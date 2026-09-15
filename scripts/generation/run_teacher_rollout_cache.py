"""Phase A: cache privileged-teacher trajectories for teacher-rollout SFT.

Each sample is routed to the bucket teacher that owns it -- exactly the routing
``ReCogDriveGoalBridgeGoalKDDistillTrainer`` applies during OPD -- and that
teacher is rolled out under the *clean* GT goal with DDIM eta=0.  Teachers are
put in ``eval()`` mode, which disables the disjoint mask/noise corruption in
``GoalCondDiffusionPlanner.encode_goal``, so what is cached is the teacher's
capability ceiling rather than a perturbed sample of it.

Cost is small because ``cache_hidden_state`` means the VLM is already frozen into
the cache: one rollout is ``num_inference_steps`` (5) forward passes of a small
DiT, no backbone involved.

Output is a single ``teacher-rollout-v1`` payload of parallel tensors (rather
than one file per token) so DataLoader workers share it copy-on-write instead of
each holding a 200k-entry dict.

Usage (see the accompanying .sh wrapper):
    torchrun --nproc_per_node=8 run_teacher_rollout_cache.py \
        --index-root .../il_training_newvlm_direct \
        --output-dir /workspace/output/tensorboard/teacher_rollout_v1 \
        --teacher-general ... --teacher-progress ... \
        --teacher-rule ... --teacher-safety ...
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers.feature_extraction_utils import BatchFeature

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config  # noqa: E402
from navsim.agents.recogdrive.recogdrive_features import (  # noqa: E402
    ReCogDriveFeatureBuilder,
    TrajectoryTargetBuilder,
)
from navsim.agents.recogdrive.recogdrive_goal_planner import GoalCondDiffusionPlanner  # noqa: E402
from navsim.agents.recogdrive.recogdrive_scene_router_agent import (  # noqa: E402
    _resolve_checkpoint_path,
    _strip_action_head_prefixes,
)
from navsim.planning.training.direct_index_dataset import DirectIndexDataset  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][rank%(rank)s] %(message)s" if False else "[%(asctime)s] %(message)s",
)
logger = logging.getLogger("teacher-rollout")

BUCKETS = (
    "general_or_no_tag",
    "progress_curbside_stopgo",
    "rule_intersection",
    "safety_dynamics_interaction",
)


# --------------------------------------------------------------------- loading
def build_teacher(
    checkpoint_path: str,
    dit_type: str,
    vlm_size: str,
    goal_mode: str,
    goal_sincos_dim: int,
    goal_hidden_dim: int,
    goal_use_heading: bool,
    name: str,
) -> GoalCondDiffusionPlanner:
    """Build one privileged teacher and verify its goal branch actually trained.

    Mirrors ``ReCogDriveSceneRouterGoalAgent._build_and_load_planner`` including
    both loud failure modes: a goal_mode mismatch silently drops projection
    weights, and an all-zero goal branch silently degrades the teacher to
    goal-free -- either would poison every cached trajectory without any error.
    """
    cfg = make_recogdrive_config(
        dit_type,
        action_dim=3,
        action_horizon=8,
        grpo=False,
        input_embedding_dim=384 if dit_type == "small" else 1536,
        sampling_method="ddim",
    )
    cfg.vlm_size = vlm_size
    planner = GoalCondDiffusionPlanner(
        cfg,
        goal_mode=goal_mode,
        goal_sincos_dim=goal_sincos_dim,
        goal_hidden_dim=goal_hidden_dim,
        goal_use_heading=goal_use_heading,
    ).cuda()

    load_kw: Dict[str, object] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw["weights_only"] = False
    resolved = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(resolved, **load_kw)
    state = checkpoint.get("state_dict", checkpoint)
    stripped = _strip_action_head_prefixes(state)

    model_goal_keys = {k for k in planner.state_dict() if "goal_" in k}
    ckpt_goal_keys = {k for k in stripped if "goal_" in k}
    if ckpt_goal_keys != model_goal_keys:
        raise RuntimeError(
            f"{name}: goal_mode={goal_mode!r} does not match {resolved!r}. "
            f"Only in checkpoint: {sorted(ckpt_goal_keys - model_goal_keys)}; "
            f"only in model: {sorted(model_goal_keys - ckpt_goal_keys)}."
        )
    dead = [k for k in sorted(ckpt_goal_keys) if stripped[k].abs().max().item() == 0.0]
    if dead:
        raise RuntimeError(
            f"{name}: {resolved!r} has all-zero goal tensors {dead} -- the goal branch "
            "never trained, so this teacher is effectively goal-free."
        )

    missing, unexpected = planner.load_state_dict(stripped, strict=False)
    logger.info(
        "loaded teacher %s (goal_mode=%s) from %s; missing=%d unexpected=%d",
        name, goal_mode, resolved, len(missing), len(unexpected),
    )
    for param in planner.parameters():
        param.requires_grad = False
    # eval() also disables goal dropout/noise inside encode_goal, so the cached
    # rollout uses the clean privileged goal.
    planner.eval()
    return planner


def collate(batch):
    features_list, targets_list, tokens = zip(*batch)
    max_len = max(f["last_hidden_state"].shape[0] for f in features_list)
    hidden = torch.zeros(
        len(features_list), max_len, features_list[0]["last_hidden_state"].shape[-1],
        dtype=features_list[0]["last_hidden_state"].dtype,
    )
    for i, f in enumerate(features_list):
        h = f["last_hidden_state"]
        hidden[i, : h.shape[0]] = h
    out = {
        "last_hidden_state": hidden,
        "history_trajectory": torch.stack([f["history_trajectory"] for f in features_list]),
        "status_feature": torch.stack([f["status_feature"] for f in features_list]),
    }
    gt = torch.stack([t["trajectory"] for t in targets_list])
    return out, gt, list(tokens)


class _SkipBadSamples(torch.utils.data.Dataset):
    """Return ``None`` for an unreadable shard instead of killing the worker."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        try:
            return self.base[idx]
        except Exception as exc:  # noqa: BLE001 - any decode/IO error is skippable
            logger.warning("skipping unreadable sample idx=%d: %r", idx, exc)
            return None


def collate_skipping(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return collate(batch)


# --------------------------------------------------------------------- rollout
@torch.no_grad()
def rollout_bucket(
    teacher: GoalCondDiffusionPlanner,
    dataset: DirectIndexDataset,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
    num_random_samples: int,
) -> Dict[str, object]:
    """Roll the teacher out over this rank's shard of ``dataset``."""
    shard = torch.utils.data.Subset(dataset, list(range(rank, len(dataset), world_size)))
    loader = DataLoader(
        _SkipBadSamples(shard),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_skipping,
        pin_memory=True,
    )
    model_dtype = next(teacher.parameters()).dtype

    tokens: List[str] = []
    trajs: List[torch.Tensor] = []
    trajs_k: List[torch.Tensor] = []
    fdes: List[torch.Tensor] = []
    ades: List[torch.Tensor] = []

    started = time.time()
    for step, batch in enumerate(loader):
        if batch is None:
            continue
        features, gt, batch_tokens = batch
        hidden = features["last_hidden_state"].cuda(non_blocking=True).to(model_dtype)
        history = features["history_trajectory"].cuda(non_blocking=True)
        status = features["status_feature"].cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True).float()

        history_flat = history.view(history.size(0), -1)
        goal = gt[:, -1, :].to(model_dtype)
        action_inputs = BatchFeature(data={
            "state": torch.cat([status, history_flat], dim=1).to(model_dtype),
            "his_traj": history_flat.to(model_dtype),
            "status_feature": status.to(model_dtype),
            "goal": goal,
        })

        pred = teacher.get_action(hidden, action_inputs, deterministic=True)["pred_traj"].float()

        if num_random_samples > 0:
            samples = []
            for _ in range(num_random_samples):
                samples.append(
                    teacher.get_action(hidden, action_inputs, deterministic=False)["pred_traj"].float()
                )
            trajs_k.append(torch.stack(samples, dim=1).cpu())

        err = (pred[..., :2] - gt[..., :2]).norm(dim=-1)
        tokens.extend(batch_tokens)
        trajs.append(pred.cpu())
        fdes.append(err[:, -1].cpu())
        ades.append(err.mean(dim=1).cpu())

        if rank == 0 and step % 50 == 0:
            done = len(tokens)
            rate = done / max(time.time() - started, 1e-6)
            logger.info(
                "  step %d: %d/%d samples on rank0 (%.1f samples/s)",
                step, done, len(shard), rate,
            )

    result = {
        "tokens": tokens,
        "trajectory": torch.cat(trajs) if trajs else torch.zeros(0, 8, 3),
        "fde_to_gt": torch.cat(fdes) if fdes else torch.zeros(0),
        "ade_to_gt": torch.cat(ades) if ades else torch.zeros(0),
    }
    if num_random_samples > 0:
        result["trajectory_samples"] = (
            torch.cat(trajs_k) if trajs_k else torch.zeros(0, num_random_samples, 8, 3)
        )
    return result


# ------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", required=True, help="il_training_newvlm_direct root")
    parser.add_argument("--index-name", default="train_index.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-general", required=True)
    parser.add_argument("--teacher-progress", required=True)
    parser.add_argument("--teacher-rule", required=True)
    parser.add_argument("--teacher-safety", required=True)
    parser.add_argument("--goal-mode", default="adaln")
    parser.add_argument("--goal-sincos-dim", type=int, default=128)
    parser.add_argument("--goal-hidden-dim", type=int, default=1024)
    parser.add_argument("--goal-use-heading", action="store_true")
    parser.add_argument("--dit-type", default="small")
    parser.add_argument("--vlm-size", default="small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--num-random-samples", type=int, default=0,
        help="extra stochastic rollouts per sample (K); 0 keeps only the deterministic one",
    )
    parser.add_argument("--limit", type=int, default=0, help="debug: cap samples per bucket")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards"
    if rank == 0:
        shard_dir.mkdir(parents=True, exist_ok=True)
    # Cheap filesystem barrier: every rank waits for rank0 to create the dir.
    for _ in range(600):
        if shard_dir.is_dir():
            break
        time.sleep(1)

    teacher_ckpts = {
        "general_or_no_tag": args.teacher_general,
        "progress_curbside_stopgo": args.teacher_progress,
        "rule_intersection": args.teacher_rule,
        "safety_dynamics_interaction": args.teacher_safety,
    }
    feature_builders = [ReCogDriveFeatureBuilder(cache_hidden_state=True, model_type="internvl")]
    target_builders = [
        TrajectoryTargetBuilder(
            trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5)
        )
    ]

    per_bucket_stats: Dict[str, Dict[str, float]] = {}
    shard: Dict[str, object] = {
        "tokens": [], "trajectory": [], "fde_to_gt": [], "ade_to_gt": [], "bucket": [],
    }
    if args.num_random_samples > 0:
        shard["trajectory_samples"] = []

    # One bucket at a time: only that bucket's teacher is resident, and the
    # per-bucket FDE it prints is directly comparable with the OPD log's
    # fde_gt_teacher_final_m.
    for bucket in BUCKETS:
        index_path = str(Path(args.index_root) / bucket / args.index_name)
        dataset = DirectIndexDataset(
            bucket_indexes=[(bucket, index_path)],
            feature_builders=feature_builders,
            target_builders=target_builders,
        )
        if args.limit > 0:
            dataset.tokens = dataset.tokens[: args.limit]
            dataset.paths = dataset.paths[: args.limit]
            dataset.buckets = dataset.buckets[: args.limit]

        if rank == 0:
            logger.info("=== bucket %s: %d samples ===", bucket, len(dataset))
        teacher = build_teacher(
            teacher_ckpts[bucket], args.dit_type, args.vlm_size, args.goal_mode,
            args.goal_sincos_dim, args.goal_hidden_dim, args.goal_use_heading, bucket,
        )
        out = rollout_bucket(
            teacher, dataset, rank, world_size,
            args.batch_size, args.num_workers, args.num_random_samples,
        )
        del teacher
        torch.cuda.empty_cache()

        shard["tokens"].extend(out["tokens"])
        shard["trajectory"].append(out["trajectory"])
        shard["fde_to_gt"].append(out["fde_to_gt"])
        shard["ade_to_gt"].append(out["ade_to_gt"])
        shard["bucket"].extend([bucket] * len(out["tokens"]))
        if args.num_random_samples > 0:
            shard["trajectory_samples"].append(out["trajectory_samples"])

        if out["fde_to_gt"].numel():
            per_bucket_stats[bucket] = {
                "rank_samples": int(out["fde_to_gt"].numel()),
                "fde_mean_m": float(out["fde_to_gt"].mean()),
                "ade_mean_m": float(out["ade_to_gt"].mean()),
            }
            if rank == 0:
                logger.info(
                    "bucket %s rank0: fde=%.3fm ade=%.3fm over %d samples",
                    bucket, per_bucket_stats[bucket]["fde_mean_m"],
                    per_bucket_stats[bucket]["ade_mean_m"], per_bucket_stats[bucket]["rank_samples"],
                )

    for key in ("trajectory", "fde_to_gt", "ade_to_gt", "trajectory_samples"):
        if key in shard:
            shard[key] = torch.cat(shard[key]) if shard[key] else torch.zeros(0)
    shard["stats"] = per_bucket_stats
    torch.save(shard, shard_dir / f"rank{rank:03d}.pt")
    logger.info("rank %d wrote %d samples", rank, len(shard["tokens"]))

    if rank != 0:
        return

    # rank0 merges once every shard has landed.
    expected = {shard_dir / f"rank{r:03d}.pt" for r in range(world_size)}
    for _ in range(3600):
        if all(p.is_file() for p in expected):
            break
        time.sleep(2)
    missing = [str(p) for p in sorted(expected) if not p.is_file()]
    if missing:
        raise RuntimeError(f"shards never appeared: {missing}")
    # A shard file can exist while its writer is still flushing; retry the load.
    shards = []
    for path in sorted(expected):
        for attempt in range(30):
            try:
                shards.append(torch.load(str(path), map_location="cpu", weights_only=False))
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 29:
                    raise RuntimeError(f"could not read shard {path}") from exc
                time.sleep(2)

    merged: Dict[str, object] = {
        "format": "teacher-rollout-v1",
        "teachers": teacher_ckpts,
        "goal_mode": args.goal_mode,
        "num_random_samples": args.num_random_samples,
        "tokens": [t for s in shards for t in s["tokens"]],
        "bucket": [b for s in shards for b in s["bucket"]],
        "trajectory": torch.cat([s["trajectory"] for s in shards]),
        "fde_to_gt": torch.cat([s["fde_to_gt"] for s in shards]),
        "ade_to_gt": torch.cat([s["ade_to_gt"] for s in shards]),
    }
    if args.num_random_samples > 0:
        merged["trajectory_samples"] = torch.cat([s["trajectory_samples"] for s in shards])

    out_path = output_dir / "teacher_rollout.pt"
    torch.save(merged, out_path)

    fde = merged["fde_to_gt"]
    summary = {
        "output": str(out_path),
        "num_samples": len(merged["tokens"]),
        "fde_mean_m": float(fde.mean()),
        "ade_mean_m": float(merged["ade_to_gt"].mean()),
        "per_bucket": {},
    }
    for bucket in BUCKETS:
        mask = torch.tensor([b == bucket for b in merged["bucket"]])
        if mask.any():
            summary["per_bucket"][bucket] = {
                "count": int(mask.sum()),
                "fde_mean_m": float(fde[mask].mean()),
                "ade_mean_m": float(merged["ade_to_gt"][mask].mean()),
            }
    for threshold in (1.0, 2.0, 3.0, 5.0):
        summary[f"frac_fde_le_{threshold:g}m"] = float((fde <= threshold).float().mean())
    (output_dir / "teacher_rollout_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("merged %d samples -> %s", summary["num_samples"], out_path)
    logger.info("%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
