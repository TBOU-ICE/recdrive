"""Phase A: offline teacher rollout cache for student SFT.

Each token is routed to its scenario-expert teacher via
``exclusive_token_to_bucket.json`` (exactly the routing the OPD trainer uses),
the teacher is conditioned on the privileged GT goal, and ``K + 1`` DDIM
rollouts are generated:

* index 0 -- the *canonical* rollout, whose ``z_T`` is seeded from the token so
  the same token always yields the same trajectory across reruns;
* indices 1..K -- additional rollouts that differ only in ``z_T``.

All rollouts use ``deterministic=True`` (DDIM eta=0).  With eta=0 the sampler is
a deterministic map ``z_T -> x0``, so drawing different ``z_T`` is the correct
way to sample the teacher's distribution; injecting extra per-step noise
(``deterministic=False``) would instead blur each sample.

Why the samples matter: the student SFT conditions on *each rollout's own
endpoint*, so K rollouts give K self-consistent ``(goal, trajectory)`` pairs per
scene and therefore K points of coverage over goal space.  The canonical rollout
is kept separate because averaging endpoints across samples would push a
regression goal head toward the mean of distinct modes.

Output: one shard per rank under ``--out-dir``, holding a stacked float32
trajectory tensor plus the token / bucket / diagnostic columns.  Nothing is
written into the (possibly read-only) feature cache.

Run with torchrun; ranks shard the token list and never communicate, so there
are no collectives to time out.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from transformers.feature_extraction_utils import BatchFeature

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402

from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config  # noqa: E402
from navsim.agents.recogdrive.recogdrive_features import (  # noqa: E402
    ReCogDriveFeatureBuilder,
    TrajectoryTargetBuilder,
)
from navsim.agents.recogdrive.recogdrive_goal_planner import GoalCondDiffusionPlanner  # noqa: E402
from navsim.agents.recogdrive.recogdrive_scene_router_agent import (  # noqa: E402
    BUCKET_NAMES,
    FALLBACK_BUCKET,
    _resolve_checkpoint_path,
    _strip_action_head_prefixes,
)
from navsim.planning.script.run_training_recogdrive_scene_router_dit_goal_distill import (  # noqa: E402
    ResilientCacheDataset,
    _load_bad_token_dirs,
    custom_collate_fn,
)
from navsim.planning.training.direct_index_dataset import (  # noqa: E402
    DirectIndexCacheDataset,
    prune_bad_tokens,
)

logger = logging.getLogger("teacher_rollout")


# --------------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--out-dir", required=True, help="Directory for the rollout shards.")
    p.add_argument("--direct-index-root", required=True,
                   help="Root holding <bucket>/train_index.json, as built by "
                        "scripts/data/prep_bucket_il_direct_indexes.py.")
    p.add_argument("--index-name", default="train_index.json",
                   help="Index file to read inside each bucket directory.")

    p.add_argument("--teacher-progress-curbside-stopgo", required=True)
    p.add_argument("--teacher-rule-intersection", required=True)
    p.add_argument("--teacher-safety-dynamics-interaction", required=True)
    p.add_argument("--teacher-general-or-no-tag", required=True)
    p.add_argument("--teacher-goal-mode", default="adaln")
    p.add_argument("--goal-sincos-dim", type=int, default=128)
    p.add_argument("--goal-hidden-dim", type=int, default=1024)
    p.add_argument("--goal-use-heading", action="store_true")

    p.add_argument("--dit-type", default="small", choices=["small", "large"])
    # String, matching the agent configs: only the literal "large" selects the
    # 3584-dim feature encoder. An int here would silently take the else branch.
    p.add_argument("--vlm-size", default="small")
    p.add_argument("--sampling-method", default="ddim")
    p.add_argument("--num-samples", type=int, default=0,
                   help="Extra rollouts per scene beyond the canonical one, each from a "
                        "different z_T. 0 keeps a single deterministic trajectory, which is "
                        "what both the teacher's own training and the OPD KD target use.")
    p.add_argument("--seed", type=int, default=0, help="Global salt for the per-token z_T seeds.")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"],
                   help="Autocast dtype; bf16 matches the OPD trainer's precision.")
    p.add_argument("--limit", type=int, default=0, help="Debug: stop after N tokens (0 = all).")
    p.add_argument("--log-every", type=int, default=50, help="Batches between progress lines.")
    return p.parse_args()


# ----------------------------------------------------------------------- teachers
def build_teacher(
    checkpoint_path_like: str,
    name: str,
    args: argparse.Namespace,
) -> GoalCondDiffusionPlanner:
    """Build one goal-conditioned teacher, rejecting a mismatched / dead goal branch.

    Mirrors ``ReCogDriveSceneRouterGoalAgent._build_and_load_planner`` +
    ``_verify_goal_weights``: a goal_mode mismatch silently drops the projection
    weights, and an all-zero goal branch silently degrades the teacher to
    goal-free.  Both must fail loudly here, because every downstream SFT label
    would otherwise be quietly wrong.
    """
    cfg = make_recogdrive_config(
        args.dit_type,
        action_dim=3,
        action_horizon=8,
        grpo=False,
        input_embedding_dim=384 if args.dit_type == "small" else 1536,
        sampling_method=args.sampling_method,
    )
    cfg.vlm_size = args.vlm_size
    planner = GoalCondDiffusionPlanner(
        cfg,
        goal_mode=args.teacher_goal_mode,
        goal_sincos_dim=args.goal_sincos_dim,
        goal_hidden_dim=args.goal_hidden_dim,
        goal_use_heading=args.goal_use_heading,
        # Corruption is training-only, but keep it off so `.train()` can never
        # silently perturb a label-generating teacher.
        goal_dropout_p=0.0,
        goal_noise_p=0.0,
    ).cuda()

    load_kw: Dict[str, object] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw["weights_only"] = False
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path_like)
    checkpoint = torch.load(checkpoint_path, **load_kw)
    state = checkpoint.get("state_dict", checkpoint)
    stripped = _strip_action_head_prefixes(state)

    model_goal_keys = {k for k in planner.state_dict() if "goal_" in k}
    ckpt_goal_keys = {k for k in stripped if "goal_" in k}
    if ckpt_goal_keys != model_goal_keys:
        raise RuntimeError(
            f"{name}: goal_mode={args.teacher_goal_mode!r} does not match checkpoint {checkpoint_path!r}. "
            f"Only in checkpoint: {sorted(ckpt_goal_keys - model_goal_keys)}; "
            f"only in model: {sorted(model_goal_keys - ckpt_goal_keys)}."
        )
    dead = [k for k in sorted(ckpt_goal_keys) if stripped[k].abs().max().item() == 0.0]
    if dead:
        raise RuntimeError(
            f"{name}: checkpoint {checkpoint_path!r} has all-zero goal tensors {dead} -- the goal "
            "branch never trained; rolling it out would produce goal-free labels."
        )

    missing, unexpected = planner.load_state_dict(stripped, strict=False)
    logger.info("loaded %s from %s (missing=%d unexpected=%d)", name, checkpoint_path, len(missing), len(unexpected))
    for param in planner.parameters():
        param.requires_grad = False
    planner.eval()
    return planner


def build_dataset(args, feature_builders, target_builders) -> DirectIndexCacheDataset:
    """Union of the four per-bucket direct indexes.

    The bucket label comes from which index file a sample was listed in, so the
    routing needs no exclusive_token_to_bucket.json: the bucket indexes already
    encode it, and they already merge navtrain with the simscale rounds.
    """
    index_paths = [str(Path(args.direct_index_root) / bucket / args.index_name) for bucket in BUCKET_NAMES]
    missing = [p for p in index_paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            "missing bucket direct index(es):\n  " + "\n  ".join(missing)
            + "\nBuild them with scripts/data/prep_bucket_il_direct_indexes.py."
        )
    data = DirectIndexCacheDataset(
        index_paths=index_paths,
        feature_builders=feature_builders,
        target_builders=target_builders,
        labels=list(BUCKET_NAMES),
    )

    bad_list_path = os.environ.get("SCENE_ROUTER_BAD_CACHE_LIST", "").strip()
    if bad_list_path and os.path.isfile(bad_list_path):
        dropped = prune_bad_tokens(data, _load_bad_token_dirs(bad_list_path))
        logger.info("pruned %d known-bad tokens via %s", dropped, bad_list_path)
    elif bad_list_path:
        logger.warning("SCENE_ROUTER_BAD_CACHE_LIST set but not found: %s", bad_list_path)
    return data


class _RankShard(torch.utils.data.Dataset):
    """Every ``world_size``-th sample, so ranks never generate the same token."""

    def __init__(self, base: torch.utils.data.Dataset, rank: int, world_size: int, limit: int = 0):
        self.base = base
        self.indices = list(range(rank, len(base), world_size))
        if limit > 0:
            self.indices = self.indices[: max(1, limit // max(world_size, 1))]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]


# ------------------------------------------------------------------------ sampling
def token_noise(
    tokens: Sequence[str],
    sample_idx: int,
    shape: Tuple[int, int],
    seed: int,
) -> torch.Tensor:
    """Per-token, per-sample ``z_T``, reproducible across runs and shardings.

    Seeding from the token (not from batch order) means a partially regenerated
    cache stays byte-identical to the original for the tokens it shares.
    """
    noise = torch.empty((len(tokens), *shape), dtype=torch.float32)
    generator = torch.Generator()
    for i, token in enumerate(tokens):
        key = f"{token}|{sample_idx}|{seed}".encode()
        generator.manual_seed(zlib.crc32(key) & 0xFFFFFFFF)
        noise[i] = torch.randn(shape, generator=generator)
    return noise


@torch.no_grad()
def rollout_batch(
    teachers: Dict[str, GoalCondDiffusionPlanner],
    buckets: List[str],
    tokens: List[str],
    last_hidden: torch.Tensor,
    history_flat: torch.Tensor,
    status: torch.Tensor,
    gt_traj: torch.Tensor,
    num_samples: int,
    seed: int,
    horizon: int,
    action_dim: int,
) -> torch.Tensor:
    """Return ``(B, num_samples + 1, horizon, action_dim)`` metric trajectories.

    Sample 0 is the canonical rollout.  Samples are generated for the whole
    batch at once per bucket, so each teacher is touched at most ``K + 1`` times
    per batch rather than once per sample.
    """
    batch_size = last_hidden.shape[0]
    device = last_hidden.device
    out = torch.zeros((batch_size, num_samples + 1, horizon, action_dim), dtype=torch.float32, device=device)

    gt_goal = gt_traj[:, -1, :]

    by_bucket: Dict[str, List[int]] = defaultdict(list)
    for i, bucket in enumerate(buckets):
        by_bucket[bucket].append(i)

    for bucket, rows in by_bucket.items():
        teacher = teachers[bucket]
        sel = torch.as_tensor(rows, dtype=torch.long, device=device)
        dtype = next(teacher.parameters()).dtype

        action_input = BatchFeature(data={
            "his_traj": history_flat.index_select(0, sel).to(dtype),
            "status_feature": status.index_select(0, sel).to(dtype),
        })
        vl = last_hidden.index_select(0, sel).to(dtype)
        goal = gt_goal.index_select(0, sel).to(dtype)
        sub_tokens = [tokens[i] for i in rows]

        for k in range(num_samples + 1):
            init = token_noise(sub_tokens, k, (horizon, action_dim), seed).to(device=device, dtype=dtype)
            with teacher.goal_context(goal):
                result = teacher.get_action(vl, action_input, init_actions=init, deterministic=True)
            out[sel, k] = result["pred_traj"].float()

    return out


# ---------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logging.basicConfig(
        level=logging.INFO,
        format=f"[rollout r{rank}] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    torch.cuda.set_device(local_rank)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teachers = {
        "progress_curbside_stopgo": args.teacher_progress_curbside_stopgo,
        "rule_intersection": args.teacher_rule_intersection,
        "safety_dynamics_interaction": args.teacher_safety_dynamics_interaction,
        "general_or_no_tag": args.teacher_general_or_no_tag,
    }
    teachers = {name: build_teacher(path, f"teacher[{name}]", args) for name, path in teachers.items()}

    feature_builders = [ReCogDriveFeatureBuilder(cache_hidden_state=True, cache_mode=False)]
    target_builders = [TrajectoryTargetBuilder(trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5))]

    data = build_dataset(args, feature_builders, target_builders)
    shard = _RankShard(data, rank, world_size, args.limit)
    logger.info("shard %d/%d: %d tokens (of %d total)", rank, world_size, len(shard), len(data))

    loader = DataLoader(
        ResilientCacheDataset(shard),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=custom_collate_fn,
    )

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]

    tokens_out: List[str] = []
    buckets_out: List[str] = []
    traj_chunks: List[torch.Tensor] = []
    fde_chunks: List[torch.Tensor] = []
    spread_chunks: List[torch.Tensor] = []
    seen: Dict[str, int] = {}
    collisions = 0

    started = time.time()
    for batch_idx, (features, targets, tokens_list) in enumerate(loader):
        last_hidden = features["last_hidden_state"].cuda(non_blocking=True)
        status = features["status_feature"].cuda(non_blocking=True)
        history = features["history_trajectory"].cuda(non_blocking=True)
        gt_traj = targets["trajectory"].cuda(non_blocking=True).float()
        history_flat = history.view(history.size(0), -1)

        # Tokens are stored exactly as the direct index lists them, which is also
        # exactly what the SFT dataloader will hand back -- no normalisation in
        # between, so the lookup can never silently miss.
        norm_tokens = [str(t) for t in tokens_list]
        buckets = [data.bucket_for_token(t) or FALLBACK_BUCKET for t in norm_tokens]

        autocast = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with autocast:
            trajectories = rollout_batch(
                teachers=teachers,
                buckets=buckets,
                tokens=norm_tokens,
                last_hidden=last_hidden,
                history_flat=history_flat,
                status=status,
                gt_traj=gt_traj,
                num_samples=args.num_samples,
                seed=args.seed,
                horizon=gt_traj.shape[1],
                action_dim=gt_traj.shape[2],
            )

        # Diagnostics: canonical-rollout FDE vs GT, and how far the K samples
        # spread apart.  A near-zero spread means K > 1 buys nothing.
        fde = (trajectories[:, 0, -1, :2] - gt_traj[:, -1, :2]).norm(dim=-1)
        if args.num_samples > 1:
            endpoints = trajectories[:, 1:, -1, :2]
            pairwise = torch.cdist(endpoints, endpoints)
            n = args.num_samples
            spread = pairwise.sum(dim=(1, 2)) / max(n * (n - 1), 1)
        else:
            spread = torch.zeros_like(fde)

        # Keep the first occurrence of each token only; a token can repeat when a
        # scene appears in more than one cache root.
        keep: List[int] = []
        for i, token in enumerate(norm_tokens):
            if not token:
                continue
            if token in seen:
                collisions += 1
                continue
            seen[token] = len(tokens_out)
            tokens_out.append(token)
            buckets_out.append(buckets[i])
            keep.append(i)

        keep_idx = torch.as_tensor(keep, dtype=torch.long, device=trajectories.device)
        if keep_idx.numel() > 0:
            traj_chunks.append(trajectories.index_select(0, keep_idx).cpu())
            fde_chunks.append(fde.index_select(0, keep_idx).cpu())
            spread_chunks.append(spread.index_select(0, keep_idx).cpu())

        if batch_idx % args.log_every == 0:
            done = len(tokens_out)
            rate = done / max(time.time() - started, 1e-6)
            logger.info(
                "batch %d | tokens %d | %.1f tok/s | fde(canon) %.3fm | spread %.3fm",
                batch_idx, done, rate, fde.mean().item(), spread.mean().item(),
            )

    if not traj_chunks:
        logger.warning("rank produced no rollouts; writing an empty shard")
        trajectories_all = torch.zeros((0, args.num_samples + 1, 8, 3), dtype=torch.float32)
        fde_all = torch.zeros((0,), dtype=torch.float32)
        spread_all = torch.zeros((0,), dtype=torch.float32)
    else:
        trajectories_all = torch.cat(traj_chunks, dim=0)
        fde_all = torch.cat(fde_chunks, dim=0)
        spread_all = torch.cat(spread_chunks, dim=0)

    assert trajectories_all.shape[0] == len(tokens_out), (
        f"row/token mismatch: {trajectories_all.shape[0]} vs {len(tokens_out)}"
    )

    shard_path = out_dir / f"teacher_rollout_shard_{rank:04d}.pt"
    torch.save(
        {
            "tokens": tokens_out,
            "buckets": buckets_out,
            "trajectories": trajectories_all,  # (N, K+1, 8, 3), index 0 = canonical
            "fde_to_gt": fde_all,              # canonical rollout vs GT endpoint
            "sample_spread": spread_all,       # mean pairwise endpoint distance over the K samples
            "meta": {
                "num_samples": args.num_samples,
                "seed": args.seed,
                "teacher_goal_mode": args.teacher_goal_mode,
                "sampling_method": args.sampling_method,
                "dit_type": args.dit_type,
                "amp": args.amp,
                "rank": rank,
                "world_size": world_size,
                "bucket_names": BUCKET_NAMES,
            },
        },
        shard_path,
    )
    logger.info("wrote %s: %d tokens (%d duplicate tokens skipped)", shard_path, len(tokens_out), collisions)

    # Per-bucket acceptance numbers.  fde_to_gt here must line up with the OPD
    # log's fde_gt_teacher_final_m; a mismatch means the routing or the goal
    # binding is wrong and nothing downstream is trustworthy.
    per_bucket: Dict[str, List[float]] = defaultdict(list)
    for bucket, value in zip(buckets_out, fde_all.tolist()):
        per_bucket[bucket].append(value)
    logger.info("--- per-bucket canonical FDE vs GT (rank %d) ---", rank)
    for bucket in BUCKET_NAMES:
        values = per_bucket.get(bucket, [])
        if values:
            mean = sum(values) / len(values)
            logger.info("  %-32s n=%7d  fde=%.3fm", bucket, len(values), mean)
        else:
            logger.info("  %-32s n=      0", bucket)
    if spread_all.numel():
        logger.info(
            "sample spread mean=%.3fm  (< 0.3m means K=%d buys little; drop to K=1)",
            spread_all.mean().item(), args.num_samples,
        )


if __name__ == "__main__":
    main()
