from typing import Tuple
from pathlib import Path
import logging
import os
import json

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule, AgentLightningDiT
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"



def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()

    if 'last_hidden_state' in features_list[0]:
        last_hidden_state = rnn_utils.pad_sequence(
            [features['last_hidden_state'] for features in features_list],
            batch_first=True,
            padding_value=0.0
        ).clone().detach()
        features = {
            'history_trajectory': history_trajectory,
            'high_command_one_hot': high_command_one_hot,
            'status_feature': status_feature,
            'last_hidden_state': last_hidden_state,
        }
    else:
        # OPD / no-hidden-state cache: pad image_path_tensor to batch
        path_tensors = [features['image_path_tensor'] for features in features_list]
        max_len = max(t.shape[0] for t in path_tensors)
        padded_paths = []
        for t in path_tensors:
            pad = max_len - t.shape[0]
            padded_paths.append(torch.nn.functional.pad(t, (0, pad), value=0))
        image_path_tensor = torch.stack(padded_paths, dim=0).cpu()
        features = {
            'history_trajectory': history_trajectory,
            'high_command_one_hot': high_command_one_hot,
            'status_feature': status_feature,
            'image_path_tensor': image_path_tensor,
        }
    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list




def maybe_override_train_logs_from_file(cfg: DictConfig) -> None:
    """
    Optional train_logs_override: YAML (root list or {train_logs: [...] }) or JSON
    with the same shape, replaces cfg.train_logs (e.g. align cache-only training with an SFT scene list).
    """
    tpl = cfg.get("train_logs_path", None)
    if tpl is None:
        return
    s = str(tpl).strip()
    if s in {"", "null", "~"}:
        return
    path = Path(s).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"train_logs_path is not an existing file: {path}")

    logs: List
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text())
        logs = data if isinstance(data, list) else data.get("train_logs")
        if not isinstance(logs, list):
            raise ValueError("train_logs JSON must be a bare list or an object with key 'train_logs'.")
    elif suffix in (".yaml", ".yml"):
        loaded = OmegaConf.load(path)
        if OmegaConf.is_list(loaded):
            logs = OmegaConf.to_object(loaded)
        elif isinstance(loaded, dict) or OmegaConf.is_config(loaded):
            entry = OmegaConf.select(loaded, "train_logs", default=None)
            if entry is None:
                raise ValueError("train_logs YAML must be a bare list root or contain key 'train_logs'.")
            logs = OmegaConf.to_object(entry)
        else:
            raise ValueError(f"Unsupported train_logs YAML shape: {type(loaded).__name__}")
    else:
        raise ValueError("train_logs_path must be .json, .yaml, or .yml")

    with open_dict(cfg):
        cfg.train_logs = ListConfig(logs)
    logger.info("Overriding train_logs from %s (%d entries)", path, len(logs))


def build_pl_logger(cfg: DictConfig):
    logger_type = str(cfg.get("logger", {}).get("type", "none")).lower()
    if logger_type in {"none", "false", "off"}:
        return False

    if logger_type == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger
        save_dir = str(cfg.get("output_dir", cfg.get("navsim_exp_root", ".")))
        experiment_name = cfg.get("experiment_name", "training")
        return TensorBoardLogger(save_dir=save_dir, name=experiment_name, version="tensorboard")

    if logger_type == "swanlab":
        try:
            from swanlab.integration.pytorch_lightning import SwanLabLogger
        except ImportError as exc:
            raise ImportError(
                "SwanLab is not installed. Install with `pip install swanlab`."
            ) from exc

        logger_cfg = cfg.get("logger", {})
        project = logger_cfg.get("project", "recdrive")
        experiment_name = logger_cfg.get("experiment_name", cfg.get("experiment_name", None))
        kwargs = {"project": project}
        if experiment_name:
            kwargs["experiment_name"] = experiment_name
        if logger_cfg.get("description", None):
            kwargs["description"] = logger_cfg.description
        return SwanLabLogger(**kwargs)

    raise ValueError(f"Unsupported logger.type={logger_type}. Use 'none', 'tensorboard', or 'swanlab'.")


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
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

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    maybe_override_train_logs_from_file(cfg)

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    meta_json = cfg.get("pretrain_meta_json")
    if meta_json not in (None, False, ""):
        ms = str(meta_json).strip()
        if ms and ms.lower() not in ("null", "~", "false"):
            from navsim.planning.script.utils.pretrain_navsim_logs import logs_from_pretrain_meta_json

            train_logs, val_logs = logs_from_pretrain_meta_json(ms)
            with open_dict(cfg):
                cfg.train_logs = ListConfig(train_logs)
                cfg.val_logs = ListConfig(val_logs)
            logger.info(
                "pretrain_meta_json=%s: %d train logs, %d val logs (from Navsim+Navsim_QA)",
                ms,
                len(train_logs),
                len(val_logs),
            )

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningDiT(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    pl_logger = build_pl_logger(cfg)
    logger.info(f"Using trainer logger: {type(pl_logger).__name__ if pl_logger else 'disabled'}")
    ckpt_root = Path(cfg.output_dir).expanduser() / "checkpoints"
    if rank == 0:
        ckpt_root.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    ckpt_every_1k = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_root),
        monitor=None,
        save_top_k=-1,
        every_n_train_steps=1000,
        filename="epoch{epoch:03d}-step{step:08d}",
        save_weights_only=False,
        save_last=True,
        enable_version_counter=False,
    )
    callbacks = [ckpt_every_1k]

    trainer = pl.Trainer(
        **cfg.trainer.params,
        logger=pl_logger,
        callbacks=callbacks,
    )

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
