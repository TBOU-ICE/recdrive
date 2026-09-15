"""Hydra-MDP++ style Extended PDM score utilities.

This module follows the single-stage Navtest EPDMS definition used by
Hydra-MDP++ (arXiv:2503.12820):

    EPDMS = NC * DAC * DDC * TL *
            (5 * TTC + 2 * C + 5 * EP + 5 * LK + 5 * EC) / 22

The original NAVSIM/RecogDrive metric cache stores the route centerline but
not the full lane-centerline objects used in the Hydra-MDP++ paper. Therefore
DDC and LK are evaluated against the cached route centerline, which is the
centerline representation available to the PDM scorer in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from shapely.geometry import Point

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    coords_array_to_polygon_array,
    state_array_to_coords_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)


@dataclass
class EPDMResults:
    """CSV-friendly record for Hydra-MDP++ style extended PDM evaluation."""

    NC: float
    DAC: float
    EP: float
    TTC: float
    C: float
    TL: float
    DDC: float
    LK: float
    EC: float
    EPDMS: float
    PDMS: float


@dataclass
class EPDMScorerConfig:
    """Thresholds reported by Hydra-MDP++ for extended rule-based metrics."""

    ddc_lk_threshold: float = 0.5  # tau_D, used by both DDC and LK
    ec_acceleration_threshold: float = 0.7  # tau_A [m/s^2]
    ec_jerk_threshold: float = 0.5  # tau_J [m/s^3]
    ec_yaw_rate_threshold: float = 0.7  # tau_Y^R [rad/s]
    ec_yaw_acceleration_threshold: float = 0.1  # tau_Y^A [rad/s^2]


def _safe_project_progress(centerline, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    shapely_points = [Point(float(x), float(y)) for x, y in points]
    return np.asarray(centerline.project(shapely_points), dtype=np.float64)


def _traffic_light_compliance(
    metric_cache: MetricCache,
    pred_states: npt.NDArray[np.float64],
    future_sampling: TrajectorySampling,
) -> float:
    """S^TL: 0 if the ego polygon crosses/intersects a red-light lane polygon."""

    vehicle_parameters = metric_cache.ego_state.car_footprint.vehicle_parameters
    ego_coords = state_array_to_coords_array(pred_states[None, ...], vehicle_parameters)
    ego_polygons = coords_array_to_polygon_array(ego_coords)[0]

    for time_idx in range(future_sampling.num_poses + 1):
        occupancy = metric_cache.observation[time_idx]
        if not any(metric_cache.observation.red_light_token in token for token in occupancy.tokens):
            continue
        intersecting = occupancy.intersects(ego_polygons[time_idx])
        if any(metric_cache.observation.red_light_token in token for token in intersecting):
            return 0.0
    return 1.0


def _driving_direction_compliance(
    metric_cache: MetricCache,
    pred_states: npt.NDArray[np.float64],
    threshold: float,
) -> float:
    """S^DDC: every consecutive motion should follow positive route-centerline direction.

    Hydra-MDP++ describes this as projecting consecutive positions onto the
    closest lane segment's positive direction and requiring the opposite-direction
    projected distance to be below tau_D. The cached route centerline is used here.
    """

    centers = pred_states[:, StateIndex.POINT]
    progress = _safe_project_progress(metric_cache.centerline, centers)
    backward_progress = np.maximum(0.0, -(progress[1:] - progress[:-1]))
    return float(np.all(backward_progress < threshold))


def _lane_keeping(metric_cache: MetricCache, pred_states: npt.NDArray[np.float64], threshold: float) -> float:
    """S^LK: every ego center should remain within tau_D of the route centerline."""

    centers = pred_states[:, StateIndex.POINT]
    distances = np.asarray(
        [metric_cache.centerline.linestring.distance(Point(float(x), float(y))) for x, y in centers],
        dtype=np.float64,
    )
    return float(np.all(distances < threshold))


def _state_dynamics_for_ec(states: npt.NDArray[np.float64], dt: float) -> Tuple[npt.NDArray[np.float64], ...]:
    """Return acceleration magnitude, jerk magnitude, yaw-rate, yaw-acceleration."""

    acceleration = np.linalg.norm(states[:, StateIndex.ACCELERATION_2D], axis=-1)
    jerk = np.gradient(acceleration, dt)
    yaw_rate = states[:, StateIndex.ANGULAR_VELOCITY]
    yaw_acceleration = states[:, StateIndex.ANGULAR_ACCELERATION]

    # Fallback for trajectories where angular derivatives are not populated.
    if np.allclose(yaw_rate, 0.0) and len(states) > 1:
        heading = np.unwrap(states[:, StateIndex.HEADING])
        yaw_rate = np.gradient(heading, dt)
    if np.allclose(yaw_acceleration, 0.0) and len(states) > 1:
        yaw_acceleration = np.gradient(yaw_rate, dt)

    return acceleration, jerk, yaw_rate, yaw_acceleration


def _rms_delta(current: npt.NDArray[np.float64], previous: npt.NDArray[np.float64]) -> float:
    length = min(len(current), len(previous))
    if length == 0:
        return 0.0
    return float(np.sqrt(np.mean((current[:length] - previous[:length]) ** 2)))


def _extended_comfort(
    current_states: npt.NDArray[np.float64],
    previous_states: Optional[npt.NDArray[np.float64]],
    current_ego_state: EgoState,
    previous_ego_state: Optional[EgoState],
    future_sampling: TrajectorySampling,
    config: EPDMScorerConfig,
) -> float:
    """S^EC following Hydra-MDP++ Eq. (11), comparing consecutive predictions."""

    if previous_states is None or previous_ego_state is None:
        # There is no preceding prediction for the first valid frame in a log.
        return 1.0

    dt = float(future_sampling.interval_length)
    frame_offset = int(round((current_ego_state.time_point.time_s - previous_ego_state.time_point.time_s) / dt))
    if frame_offset < 0 or frame_offset >= len(previous_states) - 1:
        return 1.0

    previous_projected = previous_states[frame_offset:]
    length = min(len(current_states), len(previous_projected))
    if length < 2:
        return 1.0

    current_dyn = _state_dynamics_for_ec(current_states[:length], dt)
    previous_dyn = _state_dynamics_for_ec(previous_projected[:length], dt)

    d_acc = _rms_delta(current_dyn[0], previous_dyn[0])
    d_jerk = _rms_delta(current_dyn[1], previous_dyn[1])
    d_yaw_rate = _rms_delta(current_dyn[2], previous_dyn[2])
    d_yaw_acc = _rms_delta(current_dyn[3], previous_dyn[3])

    return float(
        (d_acc <= config.ec_acceleration_threshold)
        and (d_jerk <= config.ec_jerk_threshold)
        and (d_yaw_rate <= config.ec_yaw_rate_threshold)
        and (d_yaw_acc <= config.ec_yaw_acceleration_threshold)
    )


def _simulate_predicted_states(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
) -> npt.NDArray[np.float64]:
    """Transform a model trajectory to global coordinates and run PDM LQR simulation."""

    initial_ego_state = metric_cache.ego_state
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)
    pred_states = get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point)
    return simulator.simulate_proposals(pred_states[None, ...], initial_ego_state)[0]


def epdm_score(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    previous_metric_cache: Optional[MetricCache] = None,
    previous_model_trajectory: Optional[Trajectory] = None,
    extended_config: EPDMScorerConfig = EPDMScorerConfig(),
) -> EPDMResults:
    """Run single-stage Hydra-MDP++ style EPDMS on one NAVSIM scene."""

    initial_ego_state = metric_cache.ego_state

    pdm_trajectory = metric_cache.trajectory
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)

    pdm_states, pred_states = (
        get_trajectory_as_array(pdm_trajectory, future_sampling, initial_ego_state.time_point),
        get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point),
    )
    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)
    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)

    pdm_scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )

    pred_idx = 1
    pred_sim_states = simulated_states[pred_idx]

    nc = float(scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx])
    dac = float(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx])
    ep = float(scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx])
    ttc = float(scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx])
    comfort = float(scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx])

    tl = _traffic_light_compliance(metric_cache, pred_sim_states, future_sampling)
    ddc = _driving_direction_compliance(metric_cache, pred_sim_states, extended_config.ddc_lk_threshold)
    lk = _lane_keeping(metric_cache, pred_sim_states, extended_config.ddc_lk_threshold)

    previous_sim_states = None
    previous_ego_state = None
    if previous_metric_cache is not None and previous_model_trajectory is not None:
        previous_sim_states = _simulate_predicted_states(
            previous_metric_cache,
            previous_model_trajectory,
            future_sampling,
            simulator,
        )
        previous_ego_state = previous_metric_cache.ego_state

    ec = _extended_comfort(
        pred_sim_states,
        previous_sim_states,
        metric_cache.ego_state,
        previous_ego_state,
        future_sampling,
        extended_config,
    )

    epdms = nc * dac * ddc * tl * ((5.0 * ttc + 2.0 * comfort + 5.0 * ep + 5.0 * lk + 5.0 * ec) / 22.0)

    return EPDMResults(
        NC=nc,
        DAC=dac,
        EP=ep,
        TTC=ttc,
        C=comfort,
        TL=tl,
        DDC=ddc,
        LK=lk,
        EC=ec,
        EPDMS=float(epdms),
        PDMS=float(pdm_scores[pred_idx]),
    )
