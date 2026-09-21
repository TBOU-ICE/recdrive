"""Training entry for goal-free, reverse-KL-only scene-router OPD."""

from pathlib import Path
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.script.run_training_recogdrive_scene_router_dit_goal_distill import (
    DirectIndexCacheDataset,
    ResilientCacheDataset,
    _load_bad_token_dirs,
    _prune_bad_tokens,
    custom_collate_fn,
)
from navsim.planning.training.agent_lightning_module_scene_router_kd_only import (
    AgentLightningSceneRouterKDOnly,
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
    agent.viz_output_dir = str(Path(cfg.output_dir) / "scene_router_kd_only_viz")
    lightning_module = AgentLightningSceneRouterKDOnly(agent=agent)

    train_indexes = list(cfg.get("scene_router_direct_train_indexes", []) or [])
    val_index = cfg.get("scene_router_direct_val_index", None)
    if not train_indexes or not val_index:
        raise ValueError(
            "KD-only scene-router training requires direct train and val indexes."
        )

    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()
    train_data = DirectIndexCacheDataset(
        index_paths=train_indexes,
        feature_builders=feature_builders,
        target_builders=target_builders,
    )
    val_data = DirectIndexCacheDataset(
        index_paths=[str(val_index)],
        feature_builders=feature_builders,
        target_builders=target_builders,
        log_names=cfg.val_logs,
    )

    bad_list_path = os.environ.get("SCENE_ROUTER_BAD_CACHE_LIST", "").strip()
    if bad_list_path and os.path.isfile(bad_list_path):
        bad_dirs = _load_bad_token_dirs(bad_list_path)
        dropped_train = _prune_bad_tokens(train_data, bad_dirs)
        dropped_val = _prune_bad_tokens(val_data, bad_dirs)
        logger.info(
            "Pruned %d train and %d val samples listed in %s",
            dropped_train,
            dropped_val,
            bad_list_path,
        )

    train_data = ResilientCacheDataset(train_data)
    val_data = ResilientCacheDataset(val_data)
    train_loader = DataLoader(
        train_data,
        collate_fn=custom_collate_fn,
        shuffle=True,
        **cfg.dataloader.params,
    )
    val_loader = DataLoader(
        val_data,
        collate_fn=custom_collate_fn,
        shuffle=False,
        **cfg.dataloader.params,
    )
    logger.info("Num training samples: %d", len(train_data))
    logger.info("Num validation samples: %d", len(val_data))

    checkpoint_dir = Path(str(cfg.checkpoint_dir))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor="val/loss_epoch",
        mode="min",
        save_top_k=5,
        every_n_epochs=1,
        save_last=True,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(cfg.tensorboard_dir),
        name=str(cfg.experiment_name),
    )
    trainer = pl.Trainer(
        **cfg.trainer.params,
        callbacks=[checkpoint_callback],
        logger=tensorboard_logger,
    )
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.get("ckpt_path", None) or None,
    )


if __name__ == "__main__":
    main()
