from dataclasses import asdict
import logging
import math
import os
from pathlib import Path
from types import MethodType
from typing import Dict, List, Tuple

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.utils.rnn as rnn_utils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.training.agent_lightning_module import AgentLightningDiT
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset, MixedCacheOnlyDataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _strip_cache_source(token: str) -> str:
    """MixedCacheOnlyDataset emits source-prefixed tokens; metric caches use raw tokens."""
    token = str(token)
    return token.split(":", 1)[1] if ":" in token else token


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
    return features, targets, tuple(_strip_cache_source(token) for token in tokens_list)


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


def _shaped_reward(mode: str, result_dict: Dict[str, float]) -> float:
    """Per-teacher GRPO reward shaping.

    Available train-scorer sub-metrics (PDMS / v1 scorer): NC, DAC, EP, TTC, C, DDC.
    Note the training scorer does NOT expose EPDMS-only terms (TLC/LK/HC/EC), so those
    can only be protected via the reference-policy KL, not optimized here.

    ``gate = NC * DAC`` is a hard multiplicative safety gate (collision / off-road -> 0),
    applied only to the shaping bonus so unsafe rollouts never earn scenario bonuses.

    mode="pdms" (default) reproduces the previous behaviour exactly (plain PDM score),
    so existing callers that do not set REWARD_MODE are unaffected.
    """
    nc = result_dict["no_at_fault_collisions"]
    dac = result_dict["drivable_area_compliance"]
    ep = result_dict["ego_progress"]
    ttc = result_dict["time_to_collision_within_bound"]
    ddc = result_dict["driving_direction_compliance"]
    pdms = result_dict["score"]
    gate = nc * dac

    if mode == "safety":
        return gate * (0.5 * pdms + 0.3 * ttc + 0.2 * nc)
    if mode == "rule":
        return gate * (0.6 * pdms + 0.4 * ddc)
    if mode == "progress":
        return gate * (0.6 * pdms + 0.4 * ep)
    if mode in ("general", "pdms"):
        return pdms
    raise ValueError(f"Unknown REWARD_MODE={mode!r} (expected pdms|safety|rule|progress|general)")


def install_mixed_rl_reward(agent: AbstractAgent) -> None:
    reward_mode = os.environ.get("REWARD_MODE", "pdms").strip().lower()
    logger.info("Installing GRPO reward with REWARD_MODE=%s", reward_mode)

    def reward_fn(self, pred_traj: torch.Tensor, tokens_list, cache_dict) -> torch.Tensor:
        """GRPO reward = REWARD_MODE-dependent shaping of PDM sub-metrics."""
        pred_np = pred_traj.detach().cpu().numpy()
        rewards = []
        for i, token in enumerate(tokens_list):
            raw_token = _strip_cache_source(token)
            trajectory = Trajectory(pred_np[i])
            metric_cache = cache_dict[raw_token]
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=self.simulator.proposal_sampling,
                simulator=self.simulator,
                scorer=self.train_scorer,
            )
            rewards.append(_shaped_reward(reward_mode, asdict(pdm_result)))
        return torch.tensor(rewards, device=pred_traj.device, dtype=pred_traj.dtype).detach()

    action_head = getattr(agent, "action_head", None)
    if action_head is None:
        raise RuntimeError("ReCogDrive agent does not expose action_head after initialize().")
    action_head.reward_fn = MethodType(reward_fn, action_head)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info("Global Seed set to %s", cfg.seed)
    logger.info("Path where all results are stored: %s", cfg.output_dir)

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    install_mixed_rl_reward(agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningDiT(agent=agent)

    train_sampler = None
    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert not cfg.force_cache_computation, (
            "force_cache_computation must be False when using cached data without building SceneLoader"
        )

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
            train_data = MixedCacheOnlyDataset(
                cache_paths=mixed_cache_paths,
                cache_names=mixed_cache_names,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
            )
            logger.info("Mixed cache source counts: %s", train_data.source_counts)
            if fullmix:
                logger.info(
                    "Full-mix mode: uniform shuffle over all %d cached samples; no SimScale trajectory IL loss is added.",
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
            assert cfg.cache_path is not None, "cache_path must be provided when using cached data without building SceneLoader"
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
    train_dataloader = DataLoader(
        train_data,
        collate_fn=custom_collate_fn,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **cfg.dataloader.params,
    )
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, shuffle=False, **cfg.dataloader.params)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(
        **cfg.trainer.params,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val/loss_epoch", mode="min", save_top_k=5, every_n_epochs=1)],
    )

    logger.info("Starting Training")
    trainer.fit(model=lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
