from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module_temporal_dit import AgentLightningTemporalDiT
from navsim.planning.training.temporal_cache_pair_dataset import TemporalCachePairDataset
from navsim.planning.script.run_training_recogdrive_rl import maybe_override_train_logs_from_file, build_pl_logger

logger = logging.getLogger(__name__)
CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _flatten_pair_batch(batch):
    flat = []
    for item in batch:
        sample_t, sample_next = item
        flat.append(sample_t)
        flat.append(sample_next)
    return flat


def temporal_pair_collate_fn(batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Tuple[str, ...]]:
    flat_batch = _flatten_pair_batch(batch)
    features_list, targets_list, tokens_list = zip(*flat_batch)

    history_trajectory = torch.stack([f["history_trajectory"] for f in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([f["high_command_one_hot"] for f in features_list], dim=0).cpu()
    status_feature = torch.stack([f["status_feature"] for f in features_list], dim=0).cpu()
    trajectory = torch.stack([t["trajectory"] for t in targets_list], dim=0).cpu()

    if "last_hidden_state" in features_list[0]:
        last_hidden_state = rnn_utils.pad_sequence(
            [f["last_hidden_state"] for f in features_list],
            batch_first=True,
            padding_value=0.0,
        ).clone().detach()
        features = {
            "history_trajectory": history_trajectory,
            "high_command_one_hot": high_command_one_hot,
            "status_feature": status_feature,
            "last_hidden_state": last_hidden_state,
        }
    else:
        path_tensors = [f["image_path_tensor"] for f in features_list]
        max_len = max(t.shape[0] for t in path_tensors)
        padded_paths = [torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=0) for t in path_tensors]
        features = {
            "history_trajectory": history_trajectory,
            "high_command_one_hot": high_command_one_hot,
            "status_feature": status_feature,
            "image_path_tensor": torch.stack(padded_paths, dim=0).cpu(),
        }

    targets = {"trajectory": trajectory}
    return features, targets, tokens_list


def _make_scene_loader(cfg: DictConfig, agent: AbstractAgent, logs, split_name: str) -> SceneLoader:
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if scene_filter.log_names is not None:
        scene_filter.log_names = [x for x in scene_filter.log_names if x in logs]
    else:
        scene_filter.log_names = logs
    logger.info("%s logs: %d", split_name, len(scene_filter.log_names) if scene_filter.log_names is not None else -1)
    return SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )


def build_temporal_datasets(cfg: DictConfig, agent: AbstractAgent):
    train_loader = _make_scene_loader(cfg, agent, cfg.train_logs, "train")
    val_loader = _make_scene_loader(cfg, agent, cfg.val_logs, "val")

    min_dt_s = float(cfg.get("temporal_min_dt_s", 0.1))
    max_dt_s = float(cfg.get("temporal_max_dt_s", 1.1))
    max_train_pairs = cfg.get("temporal_max_train_pairs", None)
    max_val_pairs = cfg.get("temporal_max_val_pairs", None)

    train_data = TemporalCachePairDataset(
        scene_loader=train_loader,
        cache_path=cfg.cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=cfg.train_logs,
        min_dt_s=min_dt_s,
        max_dt_s=max_dt_s,
        max_pairs=int(max_train_pairs) if max_train_pairs not in (None, "", "null") else None,
    )
    val_data = TemporalCachePairDataset(
        scene_loader=val_loader,
        cache_path=cfg.cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=cfg.val_logs,
        min_dt_s=min_dt_s,
        max_dt_s=max_dt_s,
        max_pairs=int(max_val_pairs) if max_val_pairs not in (None, "", "null") else None,
    )
    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    pl.seed_everything(cfg.seed, workers=True)
    maybe_override_train_logs_from_file(cfg)

    logger.info("Building temporal DiT distillation agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    lightning_module = AgentLightningTemporalDiT(agent=agent)
    lightning_module.debug_log_root = str(Path(cfg.output_dir).expanduser())
    lightning_module.debug_log_interval = int(cfg.get("dit_opd_debug_log_interval", 50))

    train_data, val_data = build_temporal_datasets(cfg, agent)
    logger.info("Num temporal training pairs: %d", len(train_data))
    logger.info("Num temporal validation pairs: %d", len(val_data))

    train_dataloader = DataLoader(train_data, collate_fn=temporal_pair_collate_fn, **cfg.dataloader.params, shuffle=True)
    val_dataloader = DataLoader(val_data, collate_fn=temporal_pair_collate_fn, **cfg.dataloader.params, shuffle=False)

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
            filename="temporal-epoch{epoch:03d}-step{step:08d}",
            save_weights_only=False,
            save_last=True,
            enable_version_counter=False,
        )
    ]

    trainer = pl.Trainer(**cfg.trainer.params, logger=pl_logger, callbacks=callbacks)
    trainer.fit(model=lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
