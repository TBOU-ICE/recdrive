"""Training entry for the EPDMS GRPO scene experts (v1 plan).

Data: navtrain adjacent-frame PAIRS (full + bucket-oversampled) mixed with
SimScale singles at sample-level ratios, via a custom unit batch sampler
(Lightning's distributed sampler is disabled; each rank draws with its own
seed, with replacement).

Config extras (hydra ``+epdms.*``):
  pair_table          path to navtrain_adjacent_pairs.json
  bucket_tokens       path to exclusive_<bucket>_tokens.json (navtrain)
  sim_cache_paths     list of SimScale agent-cache dirs (may be empty)
  sim_token_lists     list of token-list jsons aligned with sim_cache_paths
  ratio_nav_full      sample-level ratio of full-navtrain pair samples
  ratio_nav_bucket    sample-level ratio of bucket pair samples
  ratio_sim_bucket    sample-level ratio of SimScale singles
  steps_per_epoch     optimizer steps per epoch (fixed across experts)
"""

import logging
import os
from typing import Tuple

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module_epdms_rl import AgentLightningEpdmsRL
from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.planning.training.epdms_pair_mixed_dataset import (
    EpdmsPairMixedDataset,
    EpdmsUnitBatchSampler,
    epdms_pair_collate,
    plain_collate,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info("Output dir: %s", cfg.output_dir)

    logger.info("Building agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    epdms = cfg.epdms
    sim_cache_paths = list(OmegaConf.to_container(epdms.sim_cache_paths, resolve=True) or [])
    sim_token_lists = list(OmegaConf.to_container(epdms.sim_token_lists, resolve=True) or [])

    # Restrict training tokens to those whose reward-side metric caches exist
    # (tolerates a partially built navtrain v2 cache during smoke tests).
    from navsim.agents.recogdrive.epdms import MetricCacheIndexV2
    from navsim.common.dataloader import MetricCacheLoader
    from pathlib import Path

    nav_allowed = set(MetricCacheIndexV2(cfg.agent.epdms_metric_cache_v2_path).tokens)
    sim_allowed = None
    if sim_cache_paths:
        sim_allowed = set(MetricCacheLoader(Path(cfg.agent.metric_cache_path)).metric_cache_paths.keys())
    logger.info("reward-side availability: navtrain_v2=%d simscale_v1=%s", len(nav_allowed), len(sim_allowed or []))

    sim_manifests = list(OmegaConf.to_container(epdms.get("sim_manifests", None) or [], resolve=True)) or None
    train_data = EpdmsPairMixedDataset(
        nav_cache_path=cfg.cache_path,
        pair_table_path=epdms.pair_table,
        bucket_tokens_path=epdms.bucket_tokens,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        sim_cache_paths=sim_cache_paths,
        sim_token_list_paths=sim_token_lists,
        train_log_names=list(cfg.train_logs),
        nav_allowed_tokens=nav_allowed,
        sim_allowed_tokens=sim_allowed,
        nav_manifest=epdms.get("nav_manifest", None),
        sim_manifests=sim_manifests,
    )
    ratios = {
        "navtrain_full": float(epdms.ratio_nav_full),
        "navtrain_bucket": float(epdms.ratio_nav_bucket),
        "simscale_bucket": float(epdms.ratio_sim_bucket),
    }
    batch_sampler = EpdmsUnitBatchSampler(
        dataset=train_data,
        source_ratios=ratios,
        samples_per_batch=int(cfg.dataloader.params.batch_size),
        steps_per_epoch=int(epdms.steps_per_epoch),
        seed=int(cfg.seed),
        rank=rank,
    )
    num_workers = int(cfg.dataloader.params.num_workers)
    train_loader_kwargs = dict(
        batch_sampler=batch_sampler,
        collate_fn=epdms_pair_collate,
        num_workers=num_workers,
        pin_memory=bool(cfg.dataloader.params.get("pin_memory", True)),
    )
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = int(cfg.dataloader.params.get("prefetch_factor", 2))
        train_loader_kwargs["persistent_workers"] = bool(
            cfg.dataloader.params.get("persistent_workers", True)
        )
    train_loader = DataLoader(train_data, **train_loader_kwargs)
    logger.info(
        "Train units=%d sources=%s ratios=%s steps/epoch=%d",
        len(train_data),
        train_data.source_counts,
        ratios,
        int(epdms.steps_per_epoch),
    )

    # Full CacheOnlyDataset walks every val log on CPFS (tens of minutes). Skip for
    # smoke runs; offline PDMS/EPDMS eval is the real selection signal anyway.
    skip_val = bool(epdms.get("skip_val", False))
    limit_val = cfg.trainer.params.get("limit_val_batches", None)
    if limit_val is not None and float(limit_val) == 0.0:
        skip_val = True
    val_loader = None
    if not skip_val:
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=list(cfg.val_logs),
        )
        val_loader = DataLoader(
            val_data,
            batch_size=int(cfg.dataloader.params.batch_size),
            shuffle=False,
            collate_fn=plain_collate,
            num_workers=int(cfg.dataloader.params.num_workers),
        )
        logger.info("Num validation samples: %d", len(val_data))
    else:
        logger.info("Skipping validation dataloader (epdms.skip_val / limit_val_batches=0)")

    lightning_module = AgentLightningEpdmsRL(agent=agent)

    # Keep top-5 by val loss, checkpoint every 3 epochs (+ last). Pair with
    # trainer.params.check_val_every_n_epoch=3 so val runs on the same cadence.
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="val/loss_epoch",
        mode="min",
        save_top_k=5,
        every_n_epochs=3,
        save_last=True,
    )
    trainer = pl.Trainer(
        **cfg.trainer.params,
        use_distributed_sampler=False,
        callbacks=[checkpoint_cb],
    )

    ckpt_path = cfg.get("ckpt_path", None) or None
    if ckpt_path:
        logger.info("Resuming full trainer state from %s", ckpt_path)

    fit_kwargs = dict(
        model=lightning_module,
        train_dataloaders=train_loader,
        ckpt_path=ckpt_path,
    )
    if val_loader is not None:
        fit_kwargs["val_dataloaders"] = val_loader
    trainer.fit(**fit_kwargs)


if __name__ == "__main__":
    main()
