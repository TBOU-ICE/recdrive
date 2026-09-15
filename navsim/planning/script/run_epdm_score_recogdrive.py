from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import copy
import logging
import lzma
import os
import pickle
import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import torch
import torch.distributed as dist

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.evaluate.epdm_score import EPDMScorerConfig, epdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.metric_caching.metric_cache import MetricCache

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_epdm_score"


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size: int):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size: int, world_size: int, rank: int) -> range:
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
    if dist.get_rank() == src:
        buffer = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(buffer)).to(device)
        size_tensor = torch.tensor(len(tensor), device=device)
        dist.broadcast(size_tensor, src=src)
        dist.broadcast(tensor, src=src)
    else:
        size_tensor = torch.tensor(0, device=device)
        dist.broadcast(size_tensor, src=src)
        tensor = torch.ByteTensor(size_tensor.item()).to(device)
        dist.broadcast(tensor, src=src)
        obj = pickle.loads(tensor.cpu().numpy().tobytes())
    return obj


def _load_metric_cache(metric_cache_loader: MetricCacheLoader, token: str) -> MetricCache:
    metric_cache_path = metric_cache_loader.metric_cache_paths[token]
    with lzma.open(metric_cache_path, "rb") as f:
        return pickle.load(f)


def _build_previous_token_map(scene_loader: SceneLoader) -> Dict[str, Optional[str]]:
    previous: Dict[str, Optional[str]] = {}
    for _, tokens in scene_loader.get_tokens_list_per_log().items():
        for idx, token in enumerate(tokens):
            previous[token] = tokens[idx - 1] if idx > 0 else None
    return previous


def _compute_agent_trajectory(agent: AbstractAgent, scene_loader: SceneLoader, token: str) -> Trajectory:
    agent_input = scene_loader.get_agent_input_from_token(token)
    return agent.compute_trajectory(agent_input)


def run_epdm_score(args: List[Dict[str, Any]]) -> bytes:
    node_id = int(os.environ.get("NODE_RANK", 0))
    logger.info(f"Starting EPDMS worker, node_id={node_id}, rank={dist.get_rank()}")

    log_names = sorted({a["log_file"] for a in args})
    tokens_to_evaluate = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling, "Simulator and scorer proposal sampling has to be identical"

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    # Load all scenes for the logs handled by this rank. This keeps consecutive
    # frames available so EC can compare prediction(t-1) with prediction(t).
    all_log_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    all_log_scene_filter.log_names = log_names
    all_log_scene_filter.tokens = None
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=all_log_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )
    previous_token_map = _build_previous_token_map(scene_loader)

    local_tokens = sorted(set(tokens_to_evaluate) & set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    trajectory_cache: Dict[str, Trajectory] = {}

    epdm_results: List[Dict[str, Any]] = []
    for idx, token in enumerate(local_tokens):
        if dist.get_rank() == 0:
            logger.info(f"Rank {dist.get_rank()} processing scenario {idx + 1} / {len(local_tokens)}")

        score_row: Dict[str, Any] = {"token": token, "valid": True, "rank": dist.get_rank()}
        try:
            metric_cache = _load_metric_cache(metric_cache_loader, token)
            if token not in trajectory_cache:
                trajectory_cache[token] = _compute_agent_trajectory(agent, scene_loader, token)
            trajectory = trajectory_cache[token]

            previous_metric_cache = None
            previous_trajectory = None
            previous_token = previous_token_map.get(token)
            if previous_token in scene_loader.tokens and previous_token in metric_cache_loader.tokens:
                previous_metric_cache = _load_metric_cache(metric_cache_loader, previous_token)
                if previous_token not in trajectory_cache:
                    trajectory_cache[previous_token] = _compute_agent_trajectory(agent, scene_loader, previous_token)
                previous_trajectory = trajectory_cache[previous_token]

            result = epdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                previous_metric_cache=previous_metric_cache,
                previous_model_trajectory=previous_trajectory,
                extended_config=EPDMScorerConfig(),
            )
            score_row.update(asdict(result))
        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        epdm_results.append(score_row)

    return pickle.dumps(epdm_results)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    build_logger(cfg)
    _ = build_worker(cfg)

    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )

    if rank == 0:
        metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        tokens_to_evaluate = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
        num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
        num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
        if num_missing_metric_cache_tokens > 0:
            logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
        if num_unused_metric_cache_tokens > 0:
            logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    else:
        tokens_to_evaluate = []

    tokens_to_evaluate = broadcast_object(tokens_to_evaluate, device=device, src=0)
    logger.info("Starting Hydra-MDP++ style EPDMS scoring of %s scenarios...", str(len(tokens_to_evaluate)))

    sampler = InferenceSampler(len(tokens_to_evaluate))
    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        data_points.append({"cfg": cfg, "log_file": scene_loader.token_to_log_file[token], "tokens": [token]})

    serialized_score_rows = run_epdm_score(data_points) if len(data_points) > 0 else pickle.dumps([])
    serialized_tensor = torch.ByteTensor(list(serialized_score_rows)).to(device)

    local_size = len(serialized_tensor)
    size_list = [torch.tensor(local_size, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, torch.tensor(local_size, device=device))
    max_size = max(size_list).item()

    if local_size < max_size:
        padded_tensor = torch.cat([serialized_tensor, torch.zeros(max_size - local_size, dtype=torch.uint8, device=device)])
    else:
        padded_tensor = serialized_tensor

    gathered_results = [torch.empty_like(padded_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_results, padded_tensor)

    if dist.get_rank() == 0:
        final_results = []
        for rank_idx, gathered_tensor in enumerate(gathered_results):
            actual_size = int(size_list[rank_idx].item())
            serialized_data = gathered_tensor[:actual_size].cpu().numpy().tobytes()
            final_results.extend(pickle.loads(serialized_data))

        epdm_score_df = pd.DataFrame(final_results)
        num_successful_scenarios = int(epdm_score_df["valid"].sum())
        num_failed_scenarios = len(epdm_score_df) - num_successful_scenarios

        metric_columns = ["NC", "DAC", "EP", "TTC", "C", "TL", "DDC", "LK", "EC", "EPDMS", "PDMS"]
        average_row = epdm_score_df[metric_columns].mean(skipna=True).to_dict()
        average_row.update({"token": "average", "valid": epdm_score_df["valid"].all(), "rank": 0})
        epdm_score_df.loc[len(epdm_score_df)] = average_row

        save_path = Path(cfg.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        output_file = save_path / f"{timestamp}_epdms.csv"
        epdm_score_df.to_csv(output_file, index=False)

        logger.info(
            f"""
            Finished running Hydra-MDP++ style EPDMS evaluation.
                Number of successful scenarios: {num_successful_scenarios}.
                Number of failed scenarios: {num_failed_scenarios}.
                Final average EPDMS of valid results: {epdm_score_df.loc[epdm_score_df['token'] == 'average', 'EPDMS'].iloc[0]}.
                Results are stored in: {output_file}.
            """
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
