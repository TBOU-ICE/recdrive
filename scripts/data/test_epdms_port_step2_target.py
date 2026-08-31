"""Step 2 of the EPDMS-port parity test: reproduce the reference outputs with
the PORTED stack in this repo and compare numerically.

Run with:  PYTHONPATH=/mnt/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1 python test_epdms_port_step2_target.py
"""

import json
import sys
from pathlib import Path

import numpy as np

from navsim.agents.recogdrive.epdms import (
    MetricCacheIndexV2,
    build_v2_simulator_and_scorer,
    score_token_proposals_v2,
    two_frame_extended_comfort,
)

CACHE_ROOT = "/mnt/datasets/recdrive/20260513/nby/recdrive/metric_cache_v2"
REF_JSON = Path("/tmp/epdms_port_ref.json")

SUB_KEYS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
]


def main() -> None:
    ref = json.loads(REF_JSON.read_text())
    index = MetricCacheIndexV2(CACHE_ROOT)
    simulator, scorer = build_v2_simulator_and_scorer()

    n_bad = 0
    max_err = 0.0
    sim_states_by_token = {}
    for rec in ref["tokens"]:
        token = rec["token"]
        mc = index.load(token)
        poses = np.asarray(rec["poses"], dtype=np.float64)
        out = score_token_proposals_v2(mc, poses[None], simulator, scorer)
        sim_states_by_token[token] = out.simulated_states[0]

        sim_sum = float(np.abs(out.simulated_states[0]).sum())
        sim_err = abs(sim_sum - rec["sim_state_sum"]) / max(1.0, abs(rec["sim_state_sum"]))
        errs = {}
        for key in SUB_KEYS:
            errs[key] = abs(float(out.sub_metrics[key][0]) - rec["sub_metrics"][key])
        worst_key = max(errs, key=errs.get)
        max_err = max(max_err, errs[worst_key])
        if errs[worst_key] > 1e-6 or sim_err > 1e-9:
            n_bad += 1
            print(f"[MISMATCH] {token}: worst {worst_key}={errs[worst_key]:.3e} sim_err={sim_err:.3e}")
            for key in SUB_KEYS:
                if errs[key] > 1e-9:
                    print(f"    {key}: ported={float(out.sub_metrics[key][0]):.6f} ref={rec['sub_metrics'][key]:.6f}")

    print(f"\nsub-metric check: {len(ref['tokens'])} tokens, mismatches={n_bad}, max_abs_err={max_err:.3e}")

    ec_bad = 0
    for pair in ref["ec_pairs"]:
        cur = sim_states_by_token[pair["cur"]]
        prev = sim_states_by_token[pair["prev"]]
        ec = float(two_frame_extended_comfort(cur[None], prev, pair["dt"])[0])
        if abs(ec - pair["ec"]) > 1e-9:
            ec_bad += 1
            print(f"[EC MISMATCH] {pair['prev']} -> {pair['cur']}: ported={ec} ref={pair['ec']}")
    print(f"EC check: {len(ref['ec_pairs'])} pairs, mismatches={ec_bad}")

    if n_bad or ec_bad:
        sys.exit(1)
    print("PARITY OK")


if __name__ == "__main__":
    main()
