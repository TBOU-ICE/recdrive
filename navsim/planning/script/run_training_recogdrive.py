from typing import Tuple
from pathlib import Path
import logging
import math
import os
import random
import signal
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset, MixedCacheOnlyDataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


class ResilientCacheDataset(torch.utils.data.Dataset):
    """Resample when a cached feature is corrupt or its FUSE read stalls."""

    def __init__(self, base, max_retries: int = 8, load_timeout_s: int = 60):
        self.base = base
        self.max_retries = int(max_retries)
        self.load_timeout_s = int(load_timeout_s)

    def __len__(self):
        return len(self.base)

    def _load_one(self, idx: int):
        if self.load_timeout_s <= 0:
            return self.base[idx]

        def on_timeout(signum, frame):
            raise TimeoutError(f"cache load exceeded {self.load_timeout_s}s")

        try:
            previous = signal.signal(signal.SIGALRM, on_timeout)
        except (ValueError, OSError):
            return self.base[idx]
        try:
            signal.alarm(self.load_timeout_s)
            return self.base[idx]
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    def __getitem__(self, idx):
        size = len(self.base)
        last_error = None
        for attempt in range(self.max_retries):
            candidate = idx if attempt == 0 else random.randrange(size)
            try:
                return self._load_one(candidate)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Skipping unreadable cache sample idx=%d (attempt %d/%d): %r",
                    candidate,
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
        raise RuntimeError(
            f"Could not load a valid cache sample after {self.max_retries} retries"
        ) from last_error


def load_bad_tokens(path: str) -> set:
    """Read one '<shard path>[TAB<error>]' entry per line."""
    bad_tokens = set()
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            shard = line.split("\t", 1)[0].strip()
            if shard:
                bad_tokens.add(Path(shard).parent.name)
    return bad_tokens


def prune_bad_tokens(dataset, bad_tokens: set) -> int:
    """Remove known-bad token IDs from indexed cache datasets."""
    if not bad_tokens:
        return 0
    if isinstance(dataset, MixedCacheOnlyDataset):
        before = len(dataset.samples)
        dataset.samples = [
            sample for sample in dataset.samples if str(sample["token"]) not in bad_tokens
        ]
        dataset.tokens = [
            f"{sample['source']}:{sample['token']}" for sample in dataset.samples
        ]
        dataset.source_counts = {}
        for sample in dataset.samples:
            source = str(sample["source"])
            dataset.source_counts[source] = dataset.source_counts.get(source, 0) + 1
        return before - len(dataset.samples)
    valid = getattr(dataset, "_valid_cache_paths", None)
    if isinstance(valid, dict):
        drop = [token for token in valid if str(token) in bad_tokens]
        for token in drop:
            valid.pop(token, None)
        dataset.tokens = list(valid.keys())
        return len(drop)
    return 0




def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()

    last_hidden_state = rnn_utils.pad_sequence(
        [features['last_hidden_state'] for features in features_list],
        batch_first=True,
        padding_value=0.0
    ).clone().detach()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()

    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'last_hidden_state': last_hidden_state,
        'status_feature': status_feature
    }

    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list

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

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    train_sampler = None
    bad_list_path = os.environ.get("BAD_CACHE_LIST", "").strip()
    bad_tokens = set()
    if bad_list_path and os.path.isfile(bad_list_path):
        bad_tokens = load_bad_tokens(bad_list_path)
        logger.info("Loaded %d known-bad cache tokens from %s", len(bad_tokens), bad_list_path)
    elif bad_list_path:
        logger.warning("BAD_CACHE_LIST set but not found: %s", bad_list_path)

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"

        if cfg.get("use_mixed_cache", False):
            mixed_cache_paths = list(OmegaConf.to_container(cfg.mixed_cache.paths, resolve=True))
            mixed_cache_names = list(OmegaConf.to_container(cfg.mixed_cache.names, resolve=True))
            mixed_sample_ratios = list(OmegaConf.to_container(cfg.mixed_cache.sample_ratios, resolve=True))
            fullmix = bool(cfg.mixed_cache.get("fullmix", False))
            if fullmix:
                assert len(mixed_cache_paths) == len(mixed_cache_names), (
                    "mixed_cache.paths and mixed_cache.names must have the same length"
                )
            else:
                assert len(mixed_cache_paths) == len(mixed_cache_names) == len(mixed_sample_ratios), (
                    "mixed_cache.paths, mixed_cache.names, and mixed_cache.sample_ratios must have the same length"
                )
            logger.info("Using mixed cache training data: %s", dict(zip(mixed_cache_names, mixed_cache_paths)))
            mixed_index_path = cfg.mixed_cache.get("index_path", None) or None
            if mixed_index_path in ("", "null", "None"):
                mixed_index_path = None
            train_data = MixedCacheOnlyDataset(
                cache_paths=mixed_cache_paths,
                cache_names=mixed_cache_names,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                index_path=mixed_index_path,
            )
            dropped = prune_bad_tokens(train_data, bad_tokens)
            if dropped:
                logger.info("Pruned %d known-bad training samples", dropped)
            logger.info("Mixed cache source counts: %s", train_data.source_counts)
            if cfg.mixed_cache.get("fullmix", False):
                logger.info(
                    "Full-mix mode: uniform shuffle over all %d cached samples "
                    "(navtrain + simscale union, no ratio rebalancing)",
                    len(train_data),
                )
            else:
                sample_weights = train_data.get_sample_weights(mixed_sample_ratios)
                global_num_samples = int(cfg.mixed_cache.get("num_samples", 0)) or len(train_data)
                num_samples = int(math.ceil(global_num_samples / world_size))
                sampler_generator = torch.Generator()
                sampler_generator.manual_seed(int(cfg.seed) + rank)
                train_sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=num_samples,
                    replacement=True,
                    generator=sampler_generator,
                )
                logger.info(
                    "Mixed cache sampler ratios=%s global_num_samples=%d num_samples_per_rank=%d",
                    mixed_sample_ratios,
                    global_num_samples,
                    num_samples,
                )
        else:
            assert (
                cfg.cache_path is not None
            ), "cache_path must be provided when using cached data without building SceneLoader"
            train_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=cfg.train_logs,
            )
            dropped = prune_bad_tokens(train_data, bad_tokens)
            if dropped:
                logger.info("Pruned %d known-bad training samples", dropped)

        assert (
            cfg.cache_path is not None
        ), "cache_path must point to the navtrain cache used for validation"
        val_index_path = cfg.get("cache_index_path", None) or None
        if val_index_path in ("", "null", "None"):
            val_index_path = None
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
            index_path=val_index_path,
        )
        dropped = prune_bad_tokens(val_data, bad_tokens)
        if dropped:
            logger.info("Pruned %d known-bad validation samples", dropped)
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    if os.environ.get("RESILIENT_CACHE_LOADING", "0") == "1":
        max_retries = int(os.environ.get("CACHE_LOAD_MAX_RETRIES", "8"))
        load_timeout_s = int(os.environ.get("CACHE_LOAD_TIMEOUT_SEC", "60"))
        train_data = ResilientCacheDataset(
            train_data,
            max_retries=max_retries,
            load_timeout_s=load_timeout_s,
        )
        val_data = ResilientCacheDataset(
            val_data,
            max_retries=max_retries,
            load_timeout_s=load_timeout_s,
        )
        logger.info(
            "Enabled resilient cache loading: retries=%d timeout=%ds",
            max_retries,
            load_timeout_s,
        )

    logger.info("Building Datasets")
    if train_sampler is not None:
        train_dataloader = DataLoader(
            train_data,
            collate_fn=custom_collate_fn,
            **cfg.dataloader.params,
            sampler=train_sampler,
            shuffle=False,
        )
    else:
        train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=[pl.callbacks.ModelCheckpoint(monitor="val/loss_epoch",mode='min', save_top_k=5,every_n_epochs=1)])

    logger.info("Starting Training")
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path in ("", None):
        ckpt_path = None
    else:
        ckpt_path = str(ckpt_path)
        logger.info("Resuming Lightning training from ckpt_path=%s", ckpt_path)
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
