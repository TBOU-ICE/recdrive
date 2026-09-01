from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.planning.script.run_training_recogdrive_scene_router_dit_goal_distill import (
    ResilientCacheDataset,
    _load_bad_token_dirs,
    _prune_bad_tokens,
)

logger = logging.getLogger(__name__)
CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _normalize_token(token: object) -> Optional[str]:
    if token is None:
        return None
    if isinstance(token, (bytes, bytearray)):
        token = token.hex()
    if not isinstance(token, str):
        return None
    token = token.strip().lower()
    base, sep, suffix = token.rpartition("-")
    if not (sep and suffix.isdigit() and len(suffix) == 3 and base):
        token = token.replace("-", "")
    return token or None


def _load_allowed_tokens(path: str) -> Set[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        items = raw.keys()
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unsupported bucket token json format: {type(raw)}")
    out = {_normalize_token(x) for x in items}
    out.discard(None)
    return out


def _filter_cache_dataset(dataset: CacheOnlyDataset, allowed: Set[str]) -> CacheOnlyDataset:
    kept = {
        tok: path
        for tok, path in dataset._valid_cache_paths.items()
        if _normalize_token(tok) in allowed
    }
    dataset._valid_cache_paths = kept
    dataset.tokens = list(kept.keys())
    return dataset


def custom_collate_fn(batch):
    features_list, targets_list, tokens_list = zip(*batch)
    history_trajectory = torch.stack(
        [x["history_trajectory"] for x in features_list], dim=0
    ).cpu()
    high_command_one_hot = torch.stack(
        [x["high_command_one_hot"] for x in features_list], dim=0
    ).cpu()
    status_feature = torch.stack(
        [x["status_feature"] for x in features_list], dim=0
    ).cpu()
    last_hidden_state = rnn_utils.pad_sequence(
        [x["last_hidden_state"] for x in features_list],
        batch_first=True,
        padding_value=0.0,
    ).clone().detach()
    trajectory = torch.stack([x["trajectory"] for x in targets_list], dim=0).cpu()
    return (
        {
            "history_trajectory": history_trajectory,
            "high_command_one_hot": high_command_one_hot,
            "status_feature": status_feature,
            "last_hidden_state": last_hidden_state,
        },
        {"trajectory": trajectory},
        tokens_list,
    )


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
    lightning = AgentLightningModule(agent=agent)

    if not cfg.use_cache_without_dataset:
        raise RuntimeError("Robust goal-teacher runner currently expects cache-only training.")
    bucket_json = str(cfg.get("goal_teacher_bucket_tokens", "") or "")
    if not bucket_json or not os.path.isfile(bucket_json):
        raise RuntimeError(f"goal_teacher_bucket_tokens missing: {bucket_json!r}")
    allowed = _load_allowed_tokens(bucket_json)
    manifest = cfg.get("goal_teacher_cache_manifest", None)

    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()
    train_data = CacheOnlyDataset(
        cache_path=cfg.cache_path,
        feature_builders=feature_builders,
        target_builders=target_builders,
        log_names=cfg.train_logs,
        manifest_path=manifest,
    )
    val_data = CacheOnlyDataset(
        cache_path=cfg.cache_path,
        feature_builders=feature_builders,
        target_builders=target_builders,
        log_names=cfg.val_logs,
        manifest_path=manifest,
    )
    _filter_cache_dataset(train_data, allowed)
    _filter_cache_dataset(val_data, allowed)

    # Drop known-corrupt Alluxio shards before the loader ever touches them.
    bad_list_path = os.environ.get("SCENE_ROUTER_BAD_CACHE_LIST", "").strip()
    if bad_list_path and os.path.isfile(bad_list_path):
        bad_dirs = _load_bad_token_dirs(bad_list_path)
        n_train = _prune_bad_tokens(train_data, bad_dirs)
        n_val = _prune_bad_tokens(val_data, bad_dirs)
        logger.info(
            "Pruned known-bad shards from %s: %d bad dirs, dropped %d train + %d val tokens",
            bad_list_path,
            len(bad_dirs),
            n_train,
            n_val,
        )
    elif bad_list_path:
        logger.warning("SCENE_ROUTER_BAD_CACHE_LIST set but not found: %s", bad_list_path)

    logger.info("Robust goal teacher: %d train / %d val bucket samples", len(train_data), len(val_data))
    if len(train_data) == 0:
        raise RuntimeError("Bucket filtering produced zero training samples.")

    # Catch gzip/pickle decode errors and hung FUSE reads on any NEW bad shard.
    train_data = ResilientCacheDataset(train_data)
    val_data = ResilientCacheDataset(val_data)

    train_loader = DataLoader(train_data, collate_fn=custom_collate_fn, shuffle=True, **cfg.dataloader.params)
    val_loader = DataLoader(val_data, collate_fn=custom_collate_fn, shuffle=False, **cfg.dataloader.params)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="val/loss_epoch",
        mode="min",
        save_top_k=5,
        every_n_epochs=1,
        save_last=True,
    )
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=[checkpoint_cb])
    ckpt_path = cfg.get("ckpt_path", None) or None
    trainer.fit(lightning, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
