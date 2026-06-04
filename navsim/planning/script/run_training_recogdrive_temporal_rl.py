"""
Shared training entry-point for temporal-pair RL training (Plan A and Plan B).

Works identically to run_training_recogdrive_temporal_dit_distill.py with one
key difference: no teacher DiT is needed; the agent handles all loss/reward
logic internally.

Key differences from the regular RL training script
(run_training_recogdrive_rl.py):
  * Uses TemporalCachePairDataset + temporal_pair_collate_fn instead of
    CacheOnlyDataset, so use_cache_without_dataset must be False and
    navsim_log_path / sensor_blobs_path must be accessible.
  * Uses AgentLightningTemporalDiT which logs all scalar tensor predictions
    (reward, policy_loss, bc_loss, temporal_loss / temporal_reward_*, …).

Select Plan A or Plan B via the Hydra agent config:
  agent=recogdrive_agent_temporal_rl          # Plan A (aux loss)
  agent=recogdrive_agent_temporal_reward_rl   # Plan B (reward)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module_temporal_dit import AgentLightningTemporalDiT
from navsim.planning.training.temporal_cache_pair_dataset import TemporalCachePairDataset
from navsim.planning.script.run_training_recogdrive_temporal_dit_distill import (
    temporal_pair_collate_fn,
)
from navsim.planning.script.run_training_recogdrive_rl import (
    build_pl_logger,
    maybe_override_train_logs_from_file,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _make_scene_loader(
    cfg: DictConfig, agent: AbstractAgent, logs, split_name: str
) -> SceneLoader:
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if scene_filter.log_names is not None:
        scene_filter.log_names = [x for x in scene_filter.log_names if x in logs]
    else:
        scene_filter.log_names = list(logs)
    logger.info("%s: %d logs", split_name, len(scene_filter.log_names or []))
    return SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )


def build_temporal_datasets(cfg: DictConfig, agent: AbstractAgent):
    train_loader = _make_scene_loader(cfg, agent, cfg.train_logs, "train")
    val_loader   = _make_scene_loader(cfg, agent, cfg.val_logs,   "val")

    min_dt_s = float(cfg.get("temporal_min_dt_s", 0.1))
    max_dt_s = float(cfg.get("temporal_max_dt_s", 1.1))
    max_train = cfg.get("temporal_max_train_pairs", None)
    max_val   = cfg.get("temporal_max_val_pairs", None)

    def _to_int_or_none(v):
        return int(v) if v not in (None, "", "null") else None

    train_data = TemporalCachePairDataset(
        scene_loader=train_loader,
        cache_path=cfg.cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=cfg.train_logs,
        min_dt_s=min_dt_s,
        max_dt_s=max_dt_s,
        max_pairs=_to_int_or_none(max_train),
    )
    val_data = TemporalCachePairDataset(
        scene_loader=val_loader,
        cache_path=cfg.cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=cfg.val_logs,
        min_dt_s=min_dt_s,
        max_dt_s=max_dt_s,
        max_pairs=_to_int_or_none(max_val),
    )
    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank       = int(os.getenv("RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    pl.seed_everything(cfg.seed, workers=True)
    maybe_override_train_logs_from_file(cfg)

    logger.info("Building temporal-RL agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    lightning_module = AgentLightningTemporalDiT(agent=agent)
    lightning_module.debug_log_root    = str(Path(cfg.output_dir).expanduser())
    lightning_module.debug_log_interval = int(cfg.get("dit_opd_debug_log_interval", 50))

    logger.info("Building temporal pair datasets (requires raw NAVSIM data for pairing)")
    train_data, val_data = build_temporal_datasets(cfg, agent)
    logger.info("Temporal training pairs: %d", len(train_data))
    logger.info("Temporal validation pairs: %d", len(val_data))

    train_dl = DataLoader(
        train_data,
        collate_fn=temporal_pair_collate_fn,
        **cfg.dataloader.params,
        shuffle=True,
    )
    val_dl = DataLoader(
        val_data,
        collate_fn=temporal_pair_collate_fn,
        **cfg.dataloader.params,
        shuffle=False,
    )

    pl_logger = build_pl_logger(cfg)
    ckpt_root = Path(cfg.output_dir).expanduser() / "checkpoints"
    if rank == 0:
        ckpt_root.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_root),
            monitor=None,
            save_top_k=-1,
            every_n_train_steps=int(cfg.get("save_every_n_train_steps", 1000)),
            filename="temporal-rl-epoch{epoch:03d}-step{step:08d}",
            save_weights_only=False,
            save_last=True,
            enable_version_counter=False,
        )
    ]

    trainer = pl.Trainer(**cfg.trainer.params, logger=pl_logger, callbacks=callbacks)
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )


if __name__ == "__main__":
    main()
