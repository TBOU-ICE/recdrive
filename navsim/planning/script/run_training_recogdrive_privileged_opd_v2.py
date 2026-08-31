"""Training entry for Privileged-OPD v2 variants A/B/C/F/G."""
from __future__ import annotations

import logging
import os
from typing import List

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader

from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.planning.script.run_training_recogdrive_scene_router_dit_goal_distill import (
    ResilientCacheDataset,
    _load_allowed_tokens,
    _token_filter_cache_dataset,
    custom_collate_fn,
)
from navsim.agents.recogdrive.privileged_opd_v2.lightning import AgentLightningPrivilegedOPDV2

logger = logging.getLogger(__name__)
CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    local_rank = int(os.getenv("LOCAL_RANK", 0)); rank = int(os.getenv("RANK", 0)); world = int(os.getenv("WORLD_SIZE", 1))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)

    agent = instantiate(cfg.agent)
    agent.initialize()
    lightning = AgentLightningPrivilegedOPDV2(
        agent=agent,
        grad_diag_interval=int(cfg.get("opd_v2_grad_diag_interval", 100)),
        grad_diag_max_params=int(cfg.get("opd_v2_grad_diag_max_params", 8)),
    )
    fbs, tbs = agent.get_feature_builders(), agent.get_target_builders()
    if not cfg.use_cache_without_dataset:
        raise RuntimeError("PrivilegedOPD-v2 currently expects cache-only training")

    nav_manifest = cfg.get("opd_v2_nav_manifest", None)
    train_sets: List[torch.utils.data.Dataset] = [
        CacheOnlyDataset(cfg.cache_path, fbs, tbs, cfg.train_logs, manifest_path=nav_manifest)
    ]
    val = CacheOnlyDataset(cfg.cache_path, fbs, tbs, cfg.val_logs, manifest_path=nav_manifest)

    extra_paths = list(cfg.get("opd_v2_extra_cache_paths", []) or [])
    extra_manifests = list(cfg.get("opd_v2_extra_cache_manifests", []) or [])
    extra_token_json = list(cfg.get("opd_v2_extra_cache_token_json", []) or [])
    extra_repeats = list(cfg.get("opd_v2_extra_cache_repeats", []) or [])
    for i, path in enumerate(extra_paths):
        manifest = extra_manifests[i] if i < len(extra_manifests) else None
        ds = CacheOnlyDataset(path, fbs, tbs, None, manifest_path=manifest)
        tok_json = extra_token_json[i] if i < len(extra_token_json) else None
        if tok_json:
            _token_filter_cache_dataset(ds, _load_allowed_tokens(tok_json))
        rep = max(1, int(extra_repeats[i]) if i < len(extra_repeats) else 1)
        if len(ds):
            train_sets.extend([ds] * rep)
            logger.info("OPD extra cache %s: %d x%d", path, len(ds), rep)

    train = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    train = ResilientCacheDataset(train)
    val = ResilientCacheDataset(val)
    train_loader = DataLoader(train, shuffle=True, collate_fn=custom_collate_fn, **cfg.dataloader.params)
    val_loader = DataLoader(val, shuffle=False, collate_fn=custom_collate_fn, **cfg.dataloader.params)
    logger.info("PrivilegedOPD-v2 samples: train=%d val=%d", len(train), len(val))

    cb = pl.callbacks.ModelCheckpoint(monitor="val/loss_epoch", mode="min", save_top_k=5, save_last=True, every_n_epochs=1)
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=[cb])
    trainer.fit(lightning, train_loader, val_loader, ckpt_path=(cfg.get("ckpt_path", None) or None))


if __name__ == "__main__":
    main()
