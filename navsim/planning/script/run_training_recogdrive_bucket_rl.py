from pathlib import Path
import logging
import os
from typing import Dict, List, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module import AgentLightningDiT
from navsim.planning.script.bucket_expert_data import (
    DatasetEpochCallback,
    RatioMixedCacheDataset,
    TokenFilteredCacheOnlyDataset,
    load_token_list,
    log_dataset_summary,
    split_tokens_by_logs,
)

logger = logging.getLogger(__name__)
CONFIG_PATH = 'config/training_bucket_expert'
CONFIG_NAME = 'default_bucket_rl'


def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Tuple[str, ...]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()
    last_hidden_state = rnn_utils.pad_sequence(
        [features['last_hidden_state'] for features in features_list],
        batch_first=True,
        padding_value=0.0,
    ).clone().detach()
    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()

    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'status_feature': status_feature,
        'last_hidden_state': last_hidden_state,
    }
    targets = {'trajectory': trajectory}
    return features, targets, tokens_list


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    lightning_module = AgentLightningDiT(agent=agent)

    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()

    bucket_tokens = load_token_list(cfg.bucket.tokens_json)
    train_bucket_tokens = split_tokens_by_logs(bucket_tokens, cfg.cache_path, cfg.train_logs)
    val_bucket_tokens = split_tokens_by_logs(bucket_tokens, cfg.cache_path, cfg.val_logs)

    full_train = TokenFilteredCacheOnlyDataset(
        cache_path=cfg.cache_path,
        feature_builders=feature_builders,
        target_builders=target_builders,
        log_names=cfg.train_logs,
    )
    bucket_train = TokenFilteredCacheOnlyDataset(
        cache_path=cfg.cache_path,
        feature_builders=feature_builders,
        target_builders=target_builders,
        log_names=cfg.train_logs,
        tokens=train_bucket_tokens,
    )
    val_data = TokenFilteredCacheOnlyDataset(
        cache_path=cfg.cache_path,
        feature_builders=feature_builders,
        target_builders=target_builders,
        log_names=cfg.val_logs,
        tokens=val_bucket_tokens,
    )

    epoch_size = int(cfg.bucket.epoch_size) if cfg.bucket.epoch_size else len(full_train)
    train_data = RatioMixedCacheDataset(
        full_dataset=full_train,
        bucket_dataset=bucket_train,
        full_ratio=float(cfg.bucket.full_ratio),
        bucket_ratio=float(cfg.bucket.bucket_ratio),
        epoch_size=epoch_size,
        seed=int(cfg.seed),
    )

    log_dataset_summary(
        bucket_name=str(cfg.bucket.name),
        full_train_size=len(full_train),
        bucket_train_size=len(bucket_train),
        val_size=len(val_data),
        full_ratio=float(cfg.bucket.full_ratio),
        bucket_ratio=float(cfg.bucket.bucket_ratio),
        epoch_size=epoch_size,
    )

    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn, shuffle=True, **cfg.dataloader.params)
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, shuffle=False, **cfg.dataloader.params)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='val/loss_epoch', mode='min', save_top_k=5, every_n_epochs=1
    )
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=[checkpoint_cb, DatasetEpochCallback()])
    trainer.fit(model=lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
