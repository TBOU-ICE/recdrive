"""Stage 0 of self-distillation: full-mix goal-conditioned IL.

Produces the single checkpoint that Stage 1 uses as *both* teacher and student.
It is deliberately not a bucket expert: the per-bucket privileged teachers lost
general driving competence (82 EPDMS without a goal on general_or_no_tag, below
the 87.4 the deployed student already reaches), so specialising here would
reintroduce exactly the teacher-weaker-than-student failure self-distillation is
meant to remove.

Data is the same navtrain + SimScale-quality mixture the goal-free full-mix IL
checkpoint and the GoalBridge OPD runs consume, assembled by the shared helpers
in the scene-router runner so both stages see identical shards.  The only
difference from a plain IL run is the agent: goal-conditioned with mask/noise
corruption, so the resulting model is competent both with and without a goal.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.planning.script.run_training_recogdrive_scene_router_dit_goal_distill import (
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


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    lightning_module = AgentLightningModule(agent=agent)

    if not cfg.use_cache_without_dataset:
        raise RuntimeError("Stage-0 goal IL is cache-only; set use_cache_without_dataset=True.")
    assert not cfg.force_cache_computation
    assert cfg.cache_path is not None

    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()
    nav_manifest = cfg.get("scene_router_cache_manifest", None)
    extra_cache_manifests = list(cfg.get("scene_router_extra_cache_manifests", []) or [])

    train_datasets = [
        CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=feature_builders,
            target_builders=target_builders,
            log_names=cfg.train_logs,
            manifest_path=nav_manifest,
        )
    ]
    logger.info("navtrain cache: %d samples", len(train_datasets[0]))

    extra_cache_paths = list(cfg.get("scene_router_extra_cache_paths", []) or [])
    extra_cache_repeats = list(cfg.get("scene_router_extra_cache_repeats", []) or [])
    extra_cache_token_json = list(cfg.get("scene_router_extra_cache_token_json", []) or [])
    for i, extra_path in enumerate(extra_cache_paths):
        rep = max(int(extra_cache_repeats[i]) if i < len(extra_cache_repeats) else 1, 1)
        extra_manifest = extra_cache_manifests[i] if i < len(extra_cache_manifests) else None
        extra_ds = CacheOnlyDataset(
            cache_path=extra_path,
            feature_builders=feature_builders,
            target_builders=target_builders,
            log_names=None,
            manifest_path=extra_manifest,
        )
        token_json = extra_cache_token_json[i] if i < len(extra_cache_token_json) else None
        if token_json:
            _token_filter_cache_dataset(extra_ds, _load_allowed_tokens(token_json))
        logger.info("Extra cache %s: %d samples x repeat %d", extra_path, len(extra_ds), rep)
        if len(extra_ds) > 0:
            train_datasets.extend([extra_ds] * rep)
        else:
            logger.warning("Extra cache %s has 0 samples; skipped.", extra_path)

    train_data = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    val_data = CacheOnlyDataset(
        cache_path=cfg.cache_path,
        feature_builders=feature_builders,
        target_builders=target_builders,
        log_names=cfg.val_logs,
        manifest_path=nav_manifest,
    )

    bad_list_path = os.environ.get("SCENE_ROUTER_BAD_CACHE_LIST", "").strip()
    if bad_list_path and os.path.isfile(bad_list_path):
        bad_dirs = _load_bad_token_dirs(bad_list_path)
        n_train = _prune_bad_tokens(train_data, bad_dirs)
        n_val = _prune_bad_tokens(val_data, bad_dirs)
        logger.info("Pruned %d train + %d val tokens from %s", n_train, n_val, bad_list_path)
    elif bad_list_path:
        logger.warning("SCENE_ROUTER_BAD_CACHE_LIST set but not found: %s", bad_list_path)

    train_data = ResilientCacheDataset(train_data)
    val_data = ResilientCacheDataset(val_data)
    logger.info("Stage-0 goal IL: %d train / %d val samples", len(train_data), len(val_data))
    if len(train_data) == 0:
        raise RuntimeError("Stage-0 goal IL produced zero training samples.")

    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn, shuffle=True, **cfg.dataloader.params)
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, shuffle=False, **cfg.dataloader.params)

    tensorboard_dir = cfg.get("tensorboard_dir", None)
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(Path(cfg.output_dir) / "checkpoints") if tensorboard_dir else None,
        monitor="val/loss_epoch",
        mode="min",
        save_top_k=5,
        every_n_epochs=1,
        save_last=True,
    )
    trainer_logger = (
        TensorBoardLogger(save_dir=str(tensorboard_dir), name=str(cfg.experiment_name))
        if tensorboard_dir
        else True
    )
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=[checkpoint_cb], logger=trainer_logger)
    ckpt_path = cfg.get("ckpt_path", None) or None
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
