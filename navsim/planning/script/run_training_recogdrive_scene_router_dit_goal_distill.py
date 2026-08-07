"""Training entry for GOAL-teacher scene-router DiT OPD. Additive file."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import ConcatDataset, DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module_scene_router_goal import AgentLightningSceneRouterGoal

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


def _load_allowed_tokens(json_path: str) -> Set[str]:
    """Load a set of normalized tokens from an exclusive_token_to_bucket.json ({token: bucket})."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    keys = raw.keys() if isinstance(raw, dict) else raw
    allowed = {_normalize_token(t) for t in keys}
    allowed.discard(None)
    return allowed


def _token_filter_cache_dataset(dataset, allowed_tokens: Set[str]):
    """Restrict a CacheOnlyDataset to a token whitelist (post-filter; add-only, in-place)."""
    kept = {
        tok: path
        for tok, path in dataset._valid_cache_paths.items()
        if _normalize_token(tok) in allowed_tokens
    }
    dataset._valid_cache_paths = kept
    dataset.tokens = list(kept.keys())
    return dataset


def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Tuple[str, ...]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack([features["history_trajectory"] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features["high_command_one_hot"] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features["status_feature"] for features in features_list], dim=0).cpu()
    last_hidden_state = rnn_utils.pad_sequence(
        [features["last_hidden_state"] for features in features_list],
        batch_first=True,
        padding_value=0.0,
    ).clone().detach()
    trajectory = torch.stack([targets["trajectory"] for targets in targets_list], dim=0).cpu()

    features = {
        "history_trajectory": history_trajectory,
        "high_command_one_hot": high_command_one_hot,
        "status_feature": status_feature,
        "last_hidden_state": last_hidden_state,
    }
    targets = {"trajectory": trajectory}
    return features, targets, tokens_list


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    return (
        Dataset(
            scene_loader=train_scene_loader,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            cache_path=cfg.cache_path,
            force_cache_computation=cfg.force_cache_computation,
        ),
        Dataset(
            scene_loader=val_scene_loader,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            cache_path=cfg.cache_path,
            force_cache_computation=cfg.force_cache_computation,
        ),
    )


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info("Global Seed set to %s", cfg.seed)

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningSceneRouterGoal(agent=agent)

    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert not cfg.force_cache_computation
        assert cfg.cache_path is not None

        # navtrain full cache (filtered by train/val log split)
        train_datasets = [
            CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=feature_builders,
                target_builders=target_builders,
                log_names=cfg.train_logs,
            )
        ]
        # optional extra caches (e.g. simscale rounds). Repeat each `rep` times to
        # up-weight it (DDP-safe: ConcatDataset + shuffle, no custom sampler).
        extra_cache_paths = list(cfg.get("scene_router_extra_cache_paths", []) or [])
        extra_cache_repeats = list(cfg.get("scene_router_extra_cache_repeats", []) or [])
        extra_cache_token_json = list(cfg.get("scene_router_extra_cache_token_json", []) or [])
        for i, extra_path in enumerate(extra_cache_paths):
            rep = int(extra_cache_repeats[i]) if i < len(extra_cache_repeats) else 1
            rep = max(rep, 1)
            extra_ds = CacheOnlyDataset(
                cache_path=extra_path,
                feature_builders=feature_builders,
                target_builders=target_builders,
                log_names=None,
            )
            # token-filter to the (quality) bucket tokens for this cache, matching how
            # the experts consumed simscale (full cache linked only for quality-bucket tokens).
            token_json = extra_cache_token_json[i] if i < len(extra_cache_token_json) else None
            filtered = False
            if token_json:
                allowed = _load_allowed_tokens(token_json)
                _token_filter_cache_dataset(extra_ds, allowed)
                filtered = True
            logger.info(
                "Extra cache %s: %d samples (filtered=%s) x repeat %d",
                extra_path, len(extra_ds), filtered, rep,
            )
            if len(extra_ds) > 0:
                train_datasets.extend([extra_ds] * rep)
            else:
                logger.warning("Extra cache %s has 0 samples after filtering; skipped.", extra_path)

        train_data = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=feature_builders,
            target_builders=target_builders,
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn, shuffle=True, **cfg.dataloader.params)
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, shuffle=False, **cfg.dataloader.params)
    logger.info("Num training samples: %d", len(train_data))
    logger.info("Num validation samples: %d", len(val_data))

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="val/loss_epoch",
        mode="min",
        save_top_k=5,
        every_n_epochs=1,
        save_last=True,
    )
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=[checkpoint_cb])
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
