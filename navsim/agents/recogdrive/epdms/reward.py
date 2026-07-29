"""EPDMS reward assembly on top of the ported navsim_v2 scoring stack.

Key semantics mirrored from navsim_v2's official evaluation:

* proposals are re-simulated with the v2 ``PDMSimulator`` (LQR + bicycle);
* ego-progress is normalized **pairwise against the PDM reference proposal**
  (proposal 0), exactly like ``run_pdm_score`` which scores ``[pdm, pred]``.
  When scoring many rollouts in one batched call, the scorer's internal
  progress normalizer would couple all proposals, so we recompute the pairwise
  normalization from the scorer's raw internals;
* two-frame extended comfort (EC) is not computable inside the scorer (it needs
  the adjacent frame's plan) and is injected into the weighted average with
  weight 2 when available, or dropped from the normalization otherwise --
  identical to ``compute_final_scores`` in the official eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory

from .pdm_array_representation import ego_states_to_state_array
from .pdm_comfort_metrics import ego_is_two_frame_extended_comfort
from .pdm_enums import MultiMetricIndex, WeightedMetricIndex
from .pdm_scorer import PDMScorer, PDMScorerConfig
from .pdm_simulator import PDMSimulator

# EPDMS weighted-metric weights (v2 default config).
EPDMS_WEIGHTS = {
    "ego_progress": 5.0,
    "time_to_collision_within_bound": 5.0,
    "lane_keeping": 2.0,
    "history_comfort": 2.0,
    "two_frame_extended_comfort": 2.0,
}
MULTIPLICATIVE_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
)


def build_v2_simulator_and_scorer(
    proposal_sampling: Optional[TrajectorySampling] = None,
) -> tuple[PDMSimulator, PDMScorer]:
    """Simulator + scorer with the official EPDMS configuration."""
    proposal_sampling = proposal_sampling or TrajectorySampling(time_horizon=4, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling)
    scorer = PDMScorer(proposal_sampling, PDMScorerConfig())
    return simulator, scorer


def transform_trajectory(pred_trajectory: Trajectory, initial_ego_state: EgoState) -> InterpolatedTrajectory:
    """Ego-frame poses -> global-frame InterpolatedTrajectory (verbatim from navsim_v2)."""
    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(initial_ego_state, future_sampling.time_horizon, future_sampling.interval_length)

    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
    relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
    absolute_states = relative_to_absolute_poses(initial_ego_state.rear_axle, relative_states)

    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            [0.0, 0.0],
            [0.0, 0.0],
            timestep,
            initial_ego_state.car_footprint.vehicle_parameters,
        )
        for state, timestep in zip(absolute_states, timesteps)
    ]
    return InterpolatedTrajectory([initial_ego_state] + agent_states)


def get_trajectory_as_array(
    trajectory: InterpolatedTrajectory,
    future_sampling: TrajectorySampling,
    start_time: TimePoint,
) -> npt.NDArray[np.float64]:
    """Interpolate trajectory to the simulator sampling (verbatim from navsim_v2)."""
    times_s = np.arange(
        0.0,
        future_sampling.time_horizon + future_sampling.interval_length,
        future_sampling.interval_length,
    )
    times_s += start_time.time_s
    times_us = [int(time_s * 1e6) for time_s in times_s]
    times_us = np.clip(times_us, trajectory.start_time.time_us, trajectory.end_time.time_us)
    time_points = [TimePoint(time_us) for time_us in times_us]
    trajectory_ego_states: List[EgoState] = trajectory.get_state_at_times(time_points)
    return ego_states_to_state_array(trajectory_ego_states)


@dataclass
class TokenV2Scores:
    """Per-proposal v2 sub-metrics for one token (proposal 0 = PDM reference excluded)."""

    sub_metrics: Dict[str, npt.NDArray[np.float64]]  # each (P,)
    simulated_states: npt.NDArray[np.float64]  # (P, T, S) global frame
    time_point: TimePoint

    def epdms(self, ec: Optional[npt.NDArray[np.float64]] = None) -> npt.NDArray[np.float64]:
        return epdms_score(self.sub_metrics, ec)


def score_token_proposals_v2(
    metric_cache,
    proposals_poses: npt.NDArray[np.float64],
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> TokenV2Scores:
    """Simulate + score P ego-frame proposals (P, H, 3) for one token in one call.

    Mirrors navsim_v2's ``pdm_score``: the PDM reference trajectory is stacked as
    proposal 0 and ego-progress is re-normalized pairwise against it afterwards.
    """
    initial_ego_state = metric_cache.ego_state
    proposal_sampling = simulator.proposal_sampling

    pdm_states = get_trajectory_as_array(metric_cache.trajectory, proposal_sampling, initial_ego_state.time_point)
    pred_states = [
        get_trajectory_as_array(
            transform_trajectory(Trajectory(np.asarray(poses, dtype=np.float32)), initial_ego_state),
            proposal_sampling,
            initial_ego_state.time_point,
        )
        for poses in proposals_poses
    ]
    trajectory_states = np.stack([pdm_states] + pred_states, axis=0)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)

    scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        getattr(metric_cache, "map_parameters", None),
        None,
        getattr(metric_cache, "past_human_trajectory", None),
    )

    multi = scorer._multi_metrics  # (n_multi, P+1)
    weighted = scorer._weighted_metrics  # (n_weighted, P+1)
    progress_raw = scorer._progress_raw  # (P+1,)
    gates = multi.prod(axis=0)  # (P+1,)

    # Pairwise progress normalization against the PDM reference (proposal 0),
    # replicating the official two-proposal call semantics.
    n_prop = trajectory_states.shape[0] - 1
    ep = np.ones(n_prop, dtype=np.float64)
    ref_masked = progress_raw[0] * gates[0]
    threshold = scorer._config.progress_distance_threshold
    for i in range(n_prop):
        norm = max(ref_masked, progress_raw[i + 1] * gates[i + 1])
        if norm > threshold:
            ep[i] = float(np.clip(progress_raw[i + 1] / norm, 0.0, 1.0))

    sub = {
        "no_at_fault_collisions": multi[MultiMetricIndex.NO_COLLISION, 1:].copy(),
        "drivable_area_compliance": multi[MultiMetricIndex.DRIVABLE_AREA, 1:].copy(),
        "driving_direction_compliance": multi[MultiMetricIndex.DRIVING_DIRECTION, 1:].copy(),
        "traffic_light_compliance": multi[MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE, 1:].copy(),
        "ego_progress": ep,
        "time_to_collision_within_bound": weighted[WeightedMetricIndex.TTC, 1:].copy(),
        "lane_keeping": weighted[WeightedMetricIndex.LANE_KEEPING, 1:].copy(),
        "history_comfort": weighted[WeightedMetricIndex.HISTORY_COMFORT, 1:].copy(),
    }
    return TokenV2Scores(
        sub_metrics=sub,
        simulated_states=simulated_states[1:].copy(),
        time_point=metric_cache.timepoint,
    )


def epdms_score(
    sub_metrics: Dict[str, npt.NDArray[np.float64]],
    ec: Optional[npt.NDArray[np.float64]] = None,
) -> npt.NDArray[np.float64]:
    """EPDMS from sub-metrics; EC is injected when given, else dropped from the
    weight normalization (official NaN convention)."""
    mult = np.ones_like(np.asarray(sub_metrics["no_at_fault_collisions"], dtype=np.float64))
    for key in MULTIPLICATIVE_KEYS:
        mult = mult * np.asarray(sub_metrics[key], dtype=np.float64)

    num = np.zeros_like(mult)
    den = 0.0
    for key, weight in EPDMS_WEIGHTS.items():
        if key == "two_frame_extended_comfort":
            if ec is None:
                continue
            values = np.asarray(ec, dtype=np.float64)
        else:
            values = np.asarray(sub_metrics[key], dtype=np.float64)
        num = num + weight * values
        den += weight
    return mult * num / den


def two_frame_extended_comfort(
    current_states: npt.NDArray[np.float64],
    previous_states: npt.NDArray[np.float64],
    dt_s: float,
    interval_length: float = 0.1,
) -> npt.NDArray[np.float64]:
    """Official EC between current-frame plans and the previous frame's plan.

    :param current_states: (P, T, S) simulated states of the *current* frame.
    :param previous_states: (T, S) simulated states of the *previous* frame.
    :param dt_s: timestamp gap between the two frames (must be in (0, 0.55)).
    :return: (P,) float array of 0/1 EC outcomes.
    """
    if not (0.0 < dt_s < 0.55):
        raise ValueError(f"invalid adjacent-frame gap: {dt_s}")
    overlap_start = int(round(dt_s / interval_length))

    current = np.asarray(current_states, dtype=np.float64)
    if current.ndim == 2:
        current = current[None]
    previous = np.asarray(previous_states, dtype=np.float64)

    cur_overlap = current[:, :-overlap_start]
    prev_overlap = previous[overlap_start:][None].repeat(current.shape[0], axis=0)
    n_overlap = cur_overlap.shape[1]
    time_point_s = np.arange(n_overlap) * interval_length

    return ego_is_two_frame_extended_comfort(cur_overlap, prev_overlap, time_point_s).astype(np.float64)
