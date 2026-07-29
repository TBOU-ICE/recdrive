"""Step 1 of the EPDMS-port parity test: produce reference outputs with the
ORIGINAL navsim_v2 stack.

Run with:  PYTHONPATH=/workspace/navsim_v2 python test_epdms_port_step1_navsimv2.py

For a sample of navtest v2 metric caches it
  1) builds a deterministic ego-frame "prediction" (the cache's own PDM
     trajectory resampled at 0.5s, poses made relative to the initial state),
  2) scores it exactly like the official eval ([pdm, pred] proposals),
  3) computes two-frame extended comfort for adjacent-token pairs,
and dumps everything to JSON for step 2 to reproduce with the ported stack.
"""

import json
import lzma
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_two_frame_extended_comfort,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex

CACHE_ROOT = Path("/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/metric_cache_v2")
OUT_JSON = Path("/tmp/epdms_port_ref.json")
N_TOKENS = 40

proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
simulator = PDMSimulator(proposal_sampling)
scorer = PDMScorer(proposal_sampling, PDMScorerConfig())


def load(path: Path):
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def fake_prediction_poses(metric_cache) -> np.ndarray:
    """Deterministic ego-frame 8x0.5s poses derived from the PDM trajectory."""
    ego_state = metric_cache.ego_state
    states = get_trajectory_as_array(
        metric_cache.trajectory, TrajectorySampling(time_horizon=4, interval_length=0.5), ego_state.time_point
    )
    absolute = [StateSE2(*state[:3]) for state in states]
    relative = absolute_to_relative_poses(absolute)[1:]  # drop t=0
    poses = np.array([[p.x, p.y, p.heading] for p in relative], dtype=np.float64)
    # add a deterministic perturbation so metrics are not degenerate
    rng = np.random.RandomState(abs(hash(metric_cache.file_path.parent.name)) % (2**31))
    poses[:, :2] += rng.uniform(-0.4, 0.4, size=poses[:, :2].shape)
    poses[:, 2] += rng.uniform(-0.03, 0.03, size=poses.shape[0])
    return poses


def score_official(metric_cache, poses: np.ndarray):
    ego_state = metric_cache.ego_state
    pdm_states = get_trajectory_as_array(metric_cache.trajectory, proposal_sampling, ego_state.time_point)
    pred_traj = transform_trajectory(Trajectory(poses.astype(np.float32)), ego_state)
    pred_states = get_trajectory_as_array(pred_traj, proposal_sampling, ego_state.time_point)
    stacked = np.concatenate([pdm_states[None], pred_states[None]], axis=0)
    simulated = simulator.simulate_proposals(stacked, ego_state)
    scorer.score_proposals(
        simulated,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        getattr(metric_cache, "map_parameters", None),
        None,
        getattr(metric_cache, "past_human_trajectory", None),
    )
    multi, weighted = scorer._multi_metrics, scorer._weighted_metrics
    sub = {
        "no_at_fault_collisions": float(multi[MultiMetricIndex.NO_COLLISION, 1]),
        "drivable_area_compliance": float(multi[MultiMetricIndex.DRIVABLE_AREA, 1]),
        "driving_direction_compliance": float(multi[MultiMetricIndex.DRIVING_DIRECTION, 1]),
        "traffic_light_compliance": float(multi[MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE, 1]),
        "ego_progress": float(weighted[WeightedMetricIndex.PROGRESS, 1]),
        "time_to_collision_within_bound": float(weighted[WeightedMetricIndex.TTC, 1]),
        "lane_keeping": float(weighted[WeightedMetricIndex.LANE_KEEPING, 1]),
        "history_comfort": float(weighted[WeightedMetricIndex.HISTORY_COMFORT, 1]),
    }
    return sub, simulated[1]


def main() -> None:
    pkls = sorted(CACHE_ROOT.glob("*/*/*/metric_cache.pkl"))
    if not pkls:
        sys.exit(f"no metric caches under {CACHE_ROOT}")

    by_log = defaultdict(list)
    for p in pkls:
        by_log[p.parts[-4]].append(p)

    records, ec_records = [], []
    sim_cache = {}
    for log_name, paths in sorted(by_log.items()):
        if len(records) >= N_TOKENS:
            break
        caches = []
        for p in paths[:12]:
            mc = load(p)
            caches.append((p.parent.name, mc))
        caches.sort(key=lambda kv: kv[1].timepoint.time_us)
        for token, mc in caches:
            if len(records) >= N_TOKENS:
                break
            poses = fake_prediction_poses(mc)
            sub, sim_states = score_official(mc, poses)
            sim_cache[token] = (mc.timepoint.time_s, sim_states)
            records.append(
                {
                    "token": token,
                    "log_name": log_name,
                    "time_s": mc.timepoint.time_s,
                    "poses": poses.tolist(),
                    "sub_metrics": sub,
                    "sim_state_sum": float(np.abs(sim_states).sum()),
                }
            )
        # adjacent pairs inside this log for the EC reference
        toks = [t for t, _ in caches if t in sim_cache]
        for prev_tok, cur_tok in zip(toks[:-1], toks[1:]):
            dt = sim_cache[cur_tok][0] - sim_cache[prev_tok][0]
            if not (0.0 < dt < 0.55):
                continue
            shift = int(round(dt / 0.1))
            cur = sim_cache[cur_tok][1][None, :-shift]
            prev = sim_cache[prev_tok][1][None, shift:]
            ec = ego_is_two_frame_extended_comfort(cur, prev, np.arange(cur.shape[1]) * 0.1)[0]
            ec_records.append({"prev": prev_tok, "cur": cur_tok, "dt": dt, "ec": float(ec)})

    OUT_JSON.write_text(json.dumps({"tokens": records, "ec_pairs": ec_records}))
    print(f"wrote {OUT_JSON}: {len(records)} tokens, {len(ec_records)} ec pairs")


if __name__ == "__main__":
    main()
