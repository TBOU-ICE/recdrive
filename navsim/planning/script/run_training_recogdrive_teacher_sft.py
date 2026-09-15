"""Training entry for teacher-rollout student SFT. Additive file.

Data comes from the same per-bucket *direct indexes* the IL bucket experts use
(``run_recogdrive_bucket_expert_il_goal_newvlm.sh``): absolute token-directory
paths with navtrain and the simscale rounds already merged. No manifest, no
token whitelist, no symlinks or hardlinks -- the original caches are read in
place, so Phase A rollouts and Phase B SFT see exactly the same scene set.
"""

import logging
import os

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from pathlib import Path

from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.recogdrive.recogdrive_scene_router_agent import BUCKET_NAMES
from navsim.planning.script.run_training_recogdrive_scene_router_dit_goal_distill import (
    ResilientCacheDataset,
    _load_bad_token_dirs,
    custom_collate_fn,
)
from navsim.planning.training.agent_lightning_module_teacher_sft import AgentLightningTeacherSFT
from navsim.planning.training.direct_index_dataset import (
    DirectIndexCacheDataset,
    prune_bad_tokens,
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

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    lightning_module = AgentLightningTeacherSFT(agent=agent)

    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()

    if not cfg.use_cache_without_dataset:
        raise ValueError("Teacher SFT requires use_cache_without_dataset=True (cached VLM hidden states).")
    assert not cfg.force_cache_computation

    # Same direct indexes the IL bucket experts train on: absolute token-dir paths
    # with navtrain and the simscale rounds already merged. No manifest, no token
    # whitelist, no links -- the original caches are read in place.
    index_root = Path(cfg.teacher_sft_direct_index_root)
    train_index_paths = [str(index_root / bucket / cfg.teacher_sft_index_name) for bucket in BUCKET_NAMES]
    missing = [p for p in train_index_paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("missing bucket direct index(es):\n  " + "\n  ".join(missing))

    train_data = DirectIndexCacheDataset(
        index_paths=train_index_paths,
        feature_builders=feature_builders,
        target_builders=target_builders,
        labels=list(BUCKET_NAMES),
    )
    val_index_path = str(index_root / cfg.teacher_sft_val_index_name)
    val_data = DirectIndexCacheDataset(
        index_paths=[val_index_path],
        feature_builders=feature_builders,
        target_builders=target_builders,
    )

    bad_list_path = os.environ.get("SCENE_ROUTER_BAD_CACHE_LIST", "").strip()
    if bad_list_path and os.path.isfile(bad_list_path):
        bad_dirs = _load_bad_token_dirs(bad_list_path)
        logger.info(
            "Pruned bad shards: %d train + %d val tokens",
            prune_bad_tokens(train_data, bad_dirs), prune_bad_tokens(val_data, bad_dirs),
        )
    elif bad_list_path:
        logger.warning("SCENE_ROUTER_BAD_CACHE_LIST set but not found: %s", bad_list_path)

    train_data = ResilientCacheDataset(train_data)
    val_data = ResilientCacheDataset(val_data)

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
    ckpt_path = cfg.get("ckpt_path", None) or None
    if ckpt_path:
        logger.info("Resuming full trainer state from %s", ckpt_path)
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
