"""End-to-end smoke test of the EPDMS RL agent on ONE mini batch.

Uses: the vit agent cache (features), the (possibly partially built) navtrain
v2 metric cache, one SimScale round-0 token with its v1 metric cache, and the
vit IL checkpoint. Verifies forward reward assembly + backward.

Run:  PYTHONPATH=/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1 python test_epdms_agent_smoke.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.recogdrive.epdms import MetricCacheIndexV2
from navsim.agents.recogdrive.recogdrive_epdms_rl_agent import ReCogDriveEpdmsRLAgent
from navsim.planning.training.epdms_pair_mixed_dataset import epdms_pair_collate, normalize_token

VIT_CACHE = Path("/workspace/datasets/simscale/20260709/new_vlm_vit_hidden_state_nav_sim")
NAV_CACHE = VIT_CACHE / "recogdrive_agent_cache_dir_train"
SIM_CACHE = VIT_CACHE / "recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-0"
V2_METRIC = "/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache_train_v2"
SIM_METRIC = "/workspace/datasets/simscale/20260709/metric_cache_synthetic_reaction_pdm_v1.0-0"
PAIR_TABLE = "/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1/data/epdms/navtrain_adjacent_pairs.json"
INIT_CKPT = (
    "/workspace/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm-vit/"
    "version_0/checkpoints/epoch=199-step=312200.ckpt"
)

FEATURE_FILES = ("recogdrive_feature.gz",)


def load_sample(token_dir: Path):
    import gzip, pickle

    features, targets = {}, {}
    for gz in token_dir.glob("*.gz"):
        with gzip.open(gz, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            (targets if k == "trajectory" else features)[k] = v
    return features, targets


def find_nav_pair(v2_tokens):
    table = json.loads(Path(PAIR_TABLE).read_text())
    nav_logs = {p.name: p for p in NAV_CACHE.iterdir()}
    token_dir = {}
    checked = 0
    for prev, cur, dt in table["pairs"]:
        if prev in v2_tokens and cur in v2_tokens:
            for log_dir in nav_logs.values():
                pd, cd = log_dir / prev, log_dir / cur
                if pd.is_dir() and cd.is_dir():
                    return prev, cur, float(dt), pd, cd
            checked += 1
            if checked > 2000:
                break
    raise SystemExit("no usable pair found (v2 cache too small yet?)")


def find_sim_token():
    import csv

    meta = Path(SIM_METRIC) / "metadata"
    csvs = sorted(meta.glob("*.csv"))
    have_metric = set()
    with csvs[0].open() as f:
        for row in csv.DictReader(f):
            p = row.get("file_name") or next(iter(row.values()))
            if p:
                have_metric.add(normalize_token(Path(p).parts[-2]))
    for log_dir in SIM_CACHE.iterdir():
        if not log_dir.is_dir():
            continue
        for token_dir in log_dir.iterdir():
            if normalize_token(token_dir.name) in have_metric:
                return normalize_token(token_dir.name), token_dir
    raise SystemExit("no simscale token with both feature cache and metric cache")


def main():
    v2 = MetricCacheIndexV2(V2_METRIC)
    print(f"v2 metric cache tokens so far: {len(v2)}")
    prev, cur, dt, prev_dir, cur_dir = find_nav_pair(set(v2.tokens))
    sim_tok, sim_dir = find_sim_token()
    print(f"pair: {prev} -> {cur} (dt={dt}s) | sim: {sim_tok}")

    pf, pt = load_sample(prev_dir)
    cf, ct = load_sample(cur_dir)
    sf, st = load_sample(sim_dir)
    units = [
        {"kind": "pair", "dt": dt, "samples": [(pf, pt, prev), (cf, ct, cur)]},
        {"kind": "single", "samples": [(sf, st, sim_tok)]},
    ]
    features, targets, tokens = epdms_pair_collate(units)
    print("batch:", {k: tuple(v.shape) if isinstance(v, torch.Tensor) else v for k, v in features.items()})

    agent = ReCogDriveEpdmsRLAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        vlm_path="",
        checkpoint_path=INIT_CKPT,
        lr=2e-5,
        grpo=True,
        metric_cache_path=SIM_METRIC,
        reference_policy_checkpoint=INIT_CKPT,
        epdms_metric_cache_v2_path=V2_METRIC,
        scene_term="ep",
        scene_term_weight=0.3,
        rl_sample_time=4,
        rl_max_epochs=15,
    )
    agent.initialize()
    agent.train()

    out = agent.forward(features, targets, tokens)
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            print(f"  {k}: {float(v):.4f}")
    out["loss"].backward()
    grads = sum(p.grad.abs().sum().item() for p in agent.action_head.parameters() if p.grad is not None)
    print(f"  grad_abs_sum: {grads:.4f}")

    opt = agent.get_optimizers()
    sched = opt["lr_scheduler"]
    lrs = []
    for _ in range(15):
        lrs.append(round(sched.get_lr()[0], 8))
        sched.step()
    print("  lr schedule:", lrs)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
