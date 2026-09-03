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
    _as_str_list,
    build_mixed_bucket_datasets,
    filter_dataset_to_metric_tokens,
    merge_metric_cache_loader,
    start_gpu_keepalive,
    stop_gpu_keepalive,
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
    keepalive = start_gpu_keepalive(local_rank)

    try:
        agent: AbstractAgent = instantiate(cfg.agent)
        agent.initialize()
        lightning_module = AgentLightningDiT(agent=agent)

        feature_builders = agent.get_feature_builders()
        target_builders = agent.get_target_builders()
        extra_metric_paths = _as_str_list(cfg.bucket.get('extra_metric_cache_paths', []))
        metric_loader = getattr(getattr(agent, 'action_head', None), 'metric_cache_loader', None)
        metric_tokens = None
        if metric_loader is not None:
            logger.info('Merging extra metric caches: %s', extra_metric_paths)
            merge_metric_cache_loader(metric_loader, extra_metric_paths)
            metric_tokens = set(metric_loader.metric_cache_paths)
            logger.info('Metric cache tokens after merge: %d', len(metric_tokens))

        logger.info('Building mixed train/val datasets from prebuilt no-goal mix indexes')
        train_data, val_data = build_mixed_bucket_datasets(cfg, feature_builders, target_builders)
        if metric_tokens is not None:
            mixed = getattr(getattr(train_data, '_inner', None), 'base', None)
            if mixed is not None:
                dropped_full = filter_dataset_to_metric_tokens(mixed.full_dataset, metric_tokens)
                dropped_bucket = filter_dataset_to_metric_tokens(mixed.bucket_dataset, metric_tokens)
                logger.info(
                    'Dropped samples without metric cache: full=%d bucket=%d remaining full=%d bucket=%d',
                    dropped_full,
                    dropped_bucket,
                    len(mixed.full_dataset),
                    len(mixed.bucket_dataset),
                )
                mixed.set_epoch(0)
            dropped_val = filter_dataset_to_metric_tokens(getattr(getattr(val_data, '_inner', None), 'base', val_data), metric_tokens)
            logger.info('Dropped val samples without metric cache: %d remaining=%d', dropped_val, len(val_data))

        train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn, shuffle=True, **cfg.dataloader.params)
        val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, shuffle=False, **cfg.dataloader.params)

        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            monitor='val/loss_epoch', mode='min', save_top_k=5, every_n_epochs=1
        )
        trainer = pl.Trainer(**cfg.trainer.params, callbacks=[checkpoint_cb, DatasetEpochCallback()])
    finally:
        stop_gpu_keepalive(keepalive)

    trainer.fit(model=lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
