"""Offline entry for the GoalBridge teacher-guidance probe. Additive file.

Answers one question: when the routed teacher is evaluated at the states the
student actually visits, is its guidance sane -- and if not, is that because the
teacher is weak, because the student's z_t has left the teacher's training
distribution, or because the chain and the teacher disagree about the goal?

This is pure inference on a single GPU.  It trains nothing, writes no
checkpoint, and shares the training entry's dataset construction by importing it
rather than copying it, so the probed batches are the ones training would see.

Usage mirrors the training script's hydra overrides, e.g.::

    python navsim/planning/script/run_goalbridge_guidance_diagnostic.py \\
        agent=recogdrive_agent_goalbridge_pred_goal_kd_opd \\
        agent.checkpoint_path=/path/to/student.ckpt \\
        +guidance.student_goal_source=pred \\
        +guidance.out_dir=/path/to/out +guidance.tag=opd_after
"""

from pathlib import Path
from typing import Any, Dict, List
import logging
import os

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Subset

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.recogdrive.goalbridge_guidance_probe import GoalBridgeGuidanceProbe
from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.planning.training.goalbridge_guidance_plot import (
    plot_bev_by_error_strata,
    plot_summary,
)

# Reuse the training entry's dataset plumbing verbatim so the diagnostic cannot
# silently drift from the batches training consumes.
from navsim.planning.script.run_training_recogdrive_scene_router_dit_goal_distill import (
    DirectIndexCacheDataset,
    ResilientCacheDataset,
    _load_allowed_tokens,
    _load_bad_token_dirs,
    _prune_bad_tokens,
    _token_filter_cache_dataset,
    custom_collate_fn,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _build_probe_dataset(cfg: DictConfig, agent: AbstractAgent):
    """Same selection logic as training, restricted to the training split."""
    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()

    direct_train_indexes = list(cfg.get("scene_router_direct_train_indexes", []) or [])
    if direct_train_indexes:
        data = DirectIndexCacheDataset(
            index_paths=direct_train_indexes,
            feature_builders=feature_builders,
            target_builders=target_builders,
        )
    else:
        datasets = [
            CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=feature_builders,
                target_builders=target_builders,
                log_names=cfg.train_logs,
                manifest_path=cfg.get("scene_router_cache_manifest", None),
            )
        ]
        extra_cache_paths = list(cfg.get("scene_router_extra_cache_paths", []) or [])
        extra_token_json = list(cfg.get("scene_router_extra_cache_token_json", []) or [])
        extra_manifests = list(cfg.get("scene_router_extra_cache_manifests", []) or [])
        for i, extra_path in enumerate(extra_cache_paths):
            extra_ds = CacheOnlyDataset(
                cache_path=extra_path,
                feature_builders=feature_builders,
                target_builders=target_builders,
                log_names=None,
                manifest_path=extra_manifests[i] if i < len(extra_manifests) else None,
            )
            if i < len(extra_token_json) and extra_token_json[i]:
                _token_filter_cache_dataset(extra_ds, _load_allowed_tokens(extra_token_json[i]))
            if len(extra_ds) > 0:
                datasets.append(extra_ds)
        data = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    bad_list_path = os.environ.get("SCENE_ROUTER_BAD_CACHE_LIST", "").strip()
    if bad_list_path and os.path.isfile(bad_list_path):
        dropped = _prune_bad_tokens(data, _load_bad_token_dirs(bad_list_path))
        logger.info("Pruned %d known-bad tokens", dropped)
    return ResilientCacheDataset(data)


def _merge(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate per-batch probe payloads along the sample axis."""
    if len(batches) == 1:
        return batches[0]
    head = batches[0]
    merged: Dict[str, Any] = {
        "conditions": head["conditions"],
        "ddim_t": head["ddim_t"],
        "clip_value": head["clip_value"],
        "gt_traj": torch.cat([b["gt_traj"] for b in batches], dim=0),
        "gt_goal": torch.cat([b["gt_goal"] for b in batches], dim=0),
        "goal_err_m": torch.cat([b["goal_err_m"] for b in batches], dim=0),
        "buckets": [x for b in batches for x in b["buckets"]],
        "tokens": (
            [x for b in batches for x in b["tokens"]]
            if head["tokens"] is not None else None
        ),
        "pred_goal": (
            torch.cat([b["pred_goal"] for b in batches], dim=0)
            if head["pred_goal"] is not None else None
        ),
    }
    steps = []
    for s, head_step in enumerate(head["steps"]):
        record: Dict[str, Any] = {
            "step": head_step["step"],
            "timestep": head_step["timestep"],
            "alpha_bar": head_step["alpha_bar"],
            "traj": {}, "clamp_frac": {}, "ade_m": {}, "fde_m": {},
        }
        for key in head["conditions"]:
            for field in ("traj", "clamp_frac", "ade_m", "fde_m"):
                record[field][key] = torch.cat(
                    [b["steps"][s][field][key] for b in batches], dim=0
                )
        # Scalars are batch-level; average them across the probed batches.
        for field in ("z_second_moment_ratio", "z_abs_max"):
            record[field] = sum(b["steps"][s][field] for b in batches) / len(batches)
        steps.append(record)
    merged["steps"] = steps
    return merged


def _balanced_bucket_subset(dataset, agent, samples_per_bucket: int, seed: int):
    """Select exactly ``samples_per_bucket`` items from every routed bucket.

    The direct-index dataset exposes its ordered token list, so balancing can be
    done before any cache tensor is loaded.  Selection is deterministic for a
    seed and round-robin interleaved so every loader batch covers the buckets.
    """
    base = getattr(dataset, "base", dataset)
    tokens = getattr(base, "tokens", None)
    if not tokens:
        raise RuntimeError(
            "Balanced guidance sampling requires a dataset with an exposed token list."
        )

    bucket_names = list(agent.scene_router_trainer.bucket_names)
    routed = [
        agent.scene_router_trainer._resolve_bucket(name)
        for name in agent._buckets_for_tokens(tokens, len(tokens))
    ]
    by_bucket = {name: [] for name in bucket_names}
    for index, bucket in enumerate(routed):
        if bucket in by_bucket:
            by_bucket[bucket].append(index)

    selected = {}
    for bucket_index, bucket in enumerate(bucket_names):
        candidates = by_bucket[bucket]
        if len(candidates) < samples_per_bucket:
            raise RuntimeError(
                f"Bucket {bucket!r} has only {len(candidates)} samples; "
                f"requested {samples_per_bucket}."
            )
        generator = torch.Generator().manual_seed(seed + 7919 * bucket_index)
        order = torch.randperm(len(candidates), generator=generator)[:samples_per_bucket]
        selected[bucket] = [candidates[int(i)] for i in order]

    # Interleave buckets so partial progress remains balanced and every normal
    # batch (batch_size divisible by four) contains equal bucket coverage.
    indices = [
        selected[bucket][sample_index]
        for sample_index in range(samples_per_bucket)
        for bucket in bucket_names
    ]
    counts = {bucket: len(selected[bucket]) for bucket in bucket_names}
    return Subset(dataset, indices), counts


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    guidance = cfg.get("guidance", {}) or {}
    out_dir = str(guidance.get("out_dir", None) or Path(cfg.output_dir) / "guidance_probe")
    tag = str(guidance.get("tag", "probe"))
    student_goal_source = str(guidance.get("student_goal_source", "pred"))
    num_batches = int(guidance.get("num_batches", 1))
    samples_per_bucket = int(guidance.get("samples_per_bucket", 0))
    seed = int(guidance.get("seed", 0))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group(backend="nccl")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if distributed:
        dist.barrier()

    logger.info(
        "Building agent (student_goal_source=%s, rank=%d/%d, cuda=%d)",
        student_goal_source, rank, world_size, local_rank,
    )
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()

    dataset = _build_probe_dataset(cfg, agent)
    logger.info("Probe dataset: %d samples", len(dataset))
    bucket_counts = None
    if samples_per_bucket > 0:
        dataset, bucket_counts = _balanced_bucket_subset(
            dataset, agent, samples_per_bucket, seed
        )
        logger.info(
            "Balanced probe dataset: %d samples, bucket counts=%s",
            len(dataset), bucket_counts,
        )
    global_num_samples = len(dataset)
    if distributed:
        # Exact strided sharding: no DistributedSampler padding/duplication.
        # With 4-way bucket interleaving and 8 ranks, the global union retains
        # exactly 64 samples per bucket while every item is evaluated once.
        local_indices = list(range(rank, global_num_samples, world_size))
        dataset = Subset(dataset, local_indices)
        logger.info(
            "Rank %d shard: %d/%d samples", rank, len(dataset), global_num_samples
        )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.dataloader.params.batch_size),
        shuffle=False,
        num_workers=int(guidance.get("num_workers", 2)),
        collate_fn=custom_collate_fn,
    )

    probe = GoalBridgeGuidanceProbe(
        agent,
        student_goal_source=student_goal_source,
        seed=seed + 100000 * rank,
    )

    payloads = []
    for i, (features, targets, tokens_list) in enumerate(loader):
        if samples_per_bucket <= 0 and i >= num_batches:
            break
        payloads.append(probe.run_batch(features, targets, tokens_list))
        total_batches = len(loader) if samples_per_bucket > 0 else num_batches
        logger.info("Rank %d probed batch %d/%d", rank, i + 1, total_batches)
    if not payloads:
        raise RuntimeError(f"Probe dataset yielded no batches on rank {rank}.")

    local_payload = _merge(payloads)
    if distributed:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local_payload, gathered, dst=0)
        if rank != 0:
            dist.barrier()
            dist.destroy_process_group()
            return
        payload = _merge(gathered)
    else:
        payload = local_payload

    payload["meta"] = {
        "tag": tag,
        "student_ckpt": str(cfg.agent.get("checkpoint_path", "")),
        "student_goal_source": student_goal_source,
        "agent": str(cfg.agent.get("_target_", "")),
        "num_batches": len(payloads),
        "num_samples": len(payload["buckets"]),
        "samples_per_bucket": samples_per_bucket,
        "bucket_counts": bucket_counts,
        "world_size": world_size,
        "seed": seed,
    }

    payload_path = os.path.join(out_dir, f"{tag}_payload.pt")
    torch.save(payload, payload_path)
    print(f"[guidance] payload -> {payload_path}")
    print(f"[guidance] {plot_summary(payload, os.path.join(out_dir, f'{tag}_summary.png'))}")
    bev_dir = os.path.join(out_dir, f"{tag}_bev_by_error")
    written = plot_bev_by_error_strata(payload, bev_dir, step_numbers=(1, 2, 3, 4))
    print(f"[guidance] per-sample BEV -> {bev_dir} ({len(written)} images)")

    # Per-step table: the numbers the conclusion should rest on.
    print("\nstep  t   sqrt(a)  |z|ratio  " + "  ".join(f"{k:>14s}" for k in payload["conditions"]))
    for record in payload["steps"]:
        ades = "  ".join(
            f"{float(record['ade_m'][k].mean()):>14.3f}" for k in payload["conditions"]
        )
        print(
            f"{record['step']:>4d}  {record['timestep']:>3d}  "
            f"{record['alpha_bar'] ** 0.5:>7.3f}  {record['z_second_moment_ratio']:>8.3f}  {ades}"
        )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
