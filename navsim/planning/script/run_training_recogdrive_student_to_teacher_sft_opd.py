"""Deterministic Phase-C comparison: SFT-init versus base-init GoalBridge OPD.

This is an add-only entrypoint.  Existing SFT/OPD jobs do not import it.

Both comparison runs consume the same ordered union of the four direct bucket
indexes.  NavTrain rows are restricted to the official training logs while all
SimScale rows are retained.  A fixed held-out 4x32 probe is evaluated before
the first optimizer step and after each epoch without perturbing training RNG.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.utils.rnn as rnn_utils
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module_scene_router_goal import (
    AgentLightningSceneRouterGoal,
)
from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.planning.training.direct_index_dataset import DirectIndexDataset
from navsim.planning.training.student_to_teacher_opd import (
    FixedOrderDistributedSampler,
    FixedProbeCallback,
    FixedProbeDataset,
    SamplerAuditCallback,
    dataset_fingerprint,
    write_run_manifest,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"
BUCKETS = (
    "general_or_no_tag",
    "progress_curbside_stopgo",
    "rule_intersection",
    "safety_dynamics_interaction",
)


def collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Tuple[str, ...]]:
    """Collate cached variable-length VLM features without changing samples."""
    features_list, targets_list, tokens_list = zip(*batch)
    features = {
        "history_trajectory": torch.stack(
            [item["history_trajectory"] for item in features_list], dim=0
        ).cpu(),
        "status_feature": torch.stack(
            [item["status_feature"] for item in features_list], dim=0
        ).cpu(),
        "last_hidden_state": rnn_utils.pad_sequence(
            [item["last_hidden_state"] for item in features_list],
            batch_first=True,
            padding_value=0.0,
        ).clone().detach(),
    }
    if "high_command_one_hot" in features_list[0]:
        features["high_command_one_hot"] = torch.stack(
            [item["high_command_one_hot"] for item in features_list], dim=0
        ).cpu()
    targets = {
        "trajectory": torch.stack(
            [item["trajectory"] for item in targets_list], dim=0
        ).cpu()
    }
    return features, targets, tokens_list


def _loader_params(cfg: DictConfig) -> Dict[str, object]:
    params = dict(cfg.dataloader.params)
    # sampler owns the global truncation/order; DataLoader must not reshuffle.
    params.pop("shuffle", None)
    params.pop("sampler", None)
    params.pop("drop_last", None)
    return params


def _assert_disjoint(train_data, val_data, probe_data) -> None:
    train_tokens = set(train_data.tokens)
    val_tokens = set(val_data.tokens)
    probe_tokens = set(probe_data.tokens)
    train_paths = {str(path) for path in train_data.paths}
    val_paths = {str(path) for path in val_data._valid_cache_paths.values()}
    probe_paths = {str(sample["path"]) for sample in probe_data.samples}
    train_logs = {path.parent.name for path in train_data.paths}
    val_logs = {path.parent.name for path in val_data._valid_cache_paths.values()}

    overlaps = {
        "train_val_token": train_tokens & val_tokens,
        "train_probe_token": train_tokens & probe_tokens,
        "train_val_path": train_paths & val_paths,
        "train_probe_path": train_paths & probe_paths,
        "train_val_log": train_logs & val_logs,
    }
    if any(overlaps.values()):
        raise RuntimeError(
            "Phase-C train/validation leakage: "
            + ", ".join(f"{name}={len(values)}" for name, values in overlaps.items())
        )
    if not probe_tokens.issubset(val_tokens):
        raise RuntimeError(
            f"Probe contains {len(probe_tokens - val_tokens)} token(s) outside validation"
        )
    logger.info(
        "Leakage check passed: train=%d, val=%d, probe=%d; "
        "train-val token/path/log overlap=0 and probe is held out",
        len(train_data), len(val_data), len(probe_data),
    )


def _assert_probe_layout(
    probe_data: FixedProbeDataset,
    *,
    per_bucket: int,
    viz_per_bucket: int,
) -> None:
    counts = {bucket: 0 for bucket in BUCKETS}
    viz_counts = {bucket: 0 for bucket in BUCKETS}
    for sample in probe_data.samples:
        bucket = str(sample["bucket"])
        if bucket not in counts:
            raise RuntimeError(f"Unexpected probe bucket: {bucket}")
        counts[bucket] += 1
        viz_counts[bucket] += int(bool(sample.get("draw_viz")))
    expected_counts = {bucket: per_bucket for bucket in BUCKETS}
    expected_viz = {bucket: viz_per_bucket for bucket in BUCKETS}
    if counts != expected_counts or viz_counts != expected_viz:
        raise RuntimeError(
            f"Fixed probe layout mismatch: counts={counts}, viz={viz_counts}; "
            f"expected {per_bucket} and {viz_per_bucket} per bucket"
        )


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    pl.seed_everything(int(cfg.seed), workers=True)

    phase_cfg = cfg.get("phase_c", {})
    required = (
        "index_root",
        "val_index_path",
        "val_cache_path",
        "probe_index_path",
        "init_mode",
        "student_checkpoint",
        "repo_root",
    )
    missing = [key for key in required if not phase_cfg.get(key)]
    if missing:
        raise ValueError(f"Missing Phase-C configuration: {missing}")

    logger.info("Building GoalBridge Goal-KD agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    lightning_module = AgentLightningSceneRouterGoal(agent=agent)
    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()

    index_root = Path(str(phase_cfg.index_root))
    index_name = str(phase_cfg.get("index_name", "train_index.json"))
    index_paths = {
        bucket: str(index_root / bucket / index_name) for bucket in BUCKETS
    }
    train_logs = {str(name) for name in cfg.train_logs}
    val_logs = {str(name) for name in cfg.val_logs}
    if not train_logs or not val_logs or train_logs & val_logs:
        raise ValueError(
            "Phase-C requires non-empty, disjoint cfg.train_logs/cfg.val_logs"
        )

    # No rollout, quality, or known-bad filtering here.  A failed cache read is
    # retried for the same token by dataset.py; it is never replaced by another
    # sample, preserving the audited order on both comparison machines.
    train_data = DirectIndexDataset(
        bucket_indexes=list(index_paths.items()),
        feature_builders=feature_builders,
        target_builders=target_builders,
        navtrain_log_names=train_logs,
    )
    val_data = CacheOnlyDataset(
        cache_path=str(phase_cfg.val_cache_path),
        feature_builders=feature_builders,
        target_builders=target_builders,
        log_names=list(cfg.val_logs),
        index_path=str(phase_cfg.val_index_path),
    )
    probe_data = FixedProbeDataset(
        index_path=str(phase_cfg.probe_index_path),
        feature_builders=feature_builders,
        target_builders=target_builders,
    )
    probe_per_bucket = int(phase_cfg.get("probe_per_bucket", 32))
    probe_viz_per_bucket = int(phase_cfg.get("probe_viz_per_bucket", 3))
    _assert_probe_layout(
        probe_data,
        per_bucket=probe_per_bucket,
        viz_per_bucket=probe_viz_per_bucket,
    )
    _assert_disjoint(train_data, val_data, probe_data)

    seed = int(cfg.seed)
    train_sampler = FixedOrderDistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=True,
        # Cover every unique sample. The sampler deterministically pads at most
        # world_size-1 rows so every DDP rank still executes the same steps.
        drop_last=False,
    )
    val_sampler = FixedOrderDistributedSampler(
        val_data,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=False,
        drop_last=False,
    )
    probe_sampler = FixedOrderDistributedSampler(
        probe_data,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=False,
        drop_last=False,
    )

    loader_params = _loader_params(cfg)
    train_loader = DataLoader(
        train_data,
        collate_fn=collate_fn,
        sampler=train_sampler,
        shuffle=False,
        drop_last=False,
        **loader_params,
    )
    val_loader = DataLoader(
        val_data,
        collate_fn=collate_fn,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        **loader_params,
    )
    probe_loader_params = dict(loader_params)
    probe_loader_params["batch_size"] = int(phase_cfg.get("probe_batch_size", 4))
    probe_loader = DataLoader(
        probe_data,
        collate_fn=collate_fn,
        sampler=probe_sampler,
        shuffle=False,
        drop_last=False,
        **probe_loader_params,
    )

    if rank == 0:
        write_run_manifest(
            output_dir=str(cfg.output_dir),
            repo_root=str(phase_cfg.repo_root),
            dataset_sha256=dataset_fingerprint(train_data),
            probe_index_path=str(phase_cfg.probe_index_path),
            init_mode=str(phase_cfg.init_mode),
            student_checkpoint=str(phase_cfg.student_checkpoint),
            seed=seed,
            world_size=world_size,
            batch_size=int(loader_params["batch_size"]),
            train_samples=len(train_data),
            sampler_drop_last=train_sampler.drop_last,
            sampler_samples_per_epoch=train_sampler.total_size,
            index_paths=index_paths,
            experiment_config={
                "goal_loss_weight": float(cfg.agent.goal_loss_weight),
                "kd_weight": float(cfg.agent.kd_weight),
                "kl_precision_clip": float(cfg.agent.kl_precision_clip),
                "student_adaln_bound": float(cfg.agent.student_adaln_bound),
                "learning_rate": float(cfg.agent.lr),
                "student_goal_mode": str(cfg.agent.student_goal_mode),
                "teacher_goal_mode": str(cfg.agent.teacher_goal_mode),
                "teacher_progress": str(cfg.agent.teacher_ckpt_progress_curbside_stopgo),
                "teacher_rule": str(cfg.agent.teacher_ckpt_rule_intersection),
                "teacher_safety": str(cfg.agent.teacher_ckpt_safety_dynamics_interaction),
                "teacher_general": str(cfg.agent.teacher_ckpt_general_or_no_tag),
                "max_epochs": int(cfg.trainer.params.max_epochs),
                "precision": str(cfg.trainer.params.precision),
                "accumulate_grad_batches": int(
                    cfg.trainer.params.accumulate_grad_batches
                ),
                "gradient_clip_val": float(cfg.trainer.params.gradient_clip_val),
                "val_samples": len(val_data),
                "probe_samples": len(probe_data),
                "probe_per_bucket": probe_per_bucket,
                "probe_viz_per_bucket": probe_viz_per_bucket,
                "probe_every_n_epochs": int(
                    phase_cfg.get("probe_every_n_epochs", 1)
                ),
                "probe_interval_steps": int(
                    phase_cfg.get("probe_interval_steps", 0)
                ),
            },
        )
    if dist.is_initialized():
        dist.barrier()

    tensorboard_dir = cfg.get("tensorboard_dir", None)
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(Path(cfg.output_dir) / "checkpoints") if tensorboard_dir else None,
        monitor="val/loss_epoch",
        mode="min",
        save_top_k=5,
        every_n_epochs=1,
        save_last=True,
    )
    callbacks = [
        checkpoint_cb,
        SamplerAuditCallback(train_sampler, str(cfg.output_dir)),
        FixedProbeCallback(
            probe_loader=probe_loader,
            probe_dataset=probe_data,
            output_dir=str(cfg.output_dir),
            every_n_epochs=int(phase_cfg.get("probe_every_n_epochs", 1)),
            interval_steps=int(phase_cfg.get("probe_interval_steps", 0)),
        ),
    ]
    trainer_logger = (
        TensorBoardLogger(save_dir=str(tensorboard_dir), name=str(cfg.experiment_name))
        if tensorboard_dir
        else True
    )
    trainer_params = dict(cfg.trainer.params)
    # The explicit sampler is part of the experiment definition.  Prevent
    # Lightning from silently replacing it with its own DistributedSampler.
    trainer_params["use_distributed_sampler"] = False
    trainer = pl.Trainer(
        **trainer_params,
        callbacks=callbacks,
        logger=trainer_logger,
    )

    ckpt_path = cfg.get("ckpt_path", None) or None
    if ckpt_path:
        logger.info("Resuming full trainer state from %s", ckpt_path)
    logger.info(
        "Phase-C init=%s train=%d (%s) val=%d probe=%d order_epoch0=%s",
        phase_cfg.init_mode,
        len(train_data),
        dataset_fingerprint(train_data),
        len(val_data),
        len(probe_data),
        train_sampler.order_sha256(0),
    )
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
