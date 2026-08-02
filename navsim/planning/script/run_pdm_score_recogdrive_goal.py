"""PDM scoring for goal-conditioned (privileged) ReCogDrive agents.

Identical to ``run_pdm_score_recogdrive.py`` except that the per-token loop sets
``requires_scene = True``, so the ``Scene`` is loaded and handed to
``ReCogDriveGoalAgent.compute_trajectory``, which reads the ground-truth goal
point from it.

Scores produced here are **privileged** (oracle) and must be reported as such --
they are the analogue of the row marked with a dagger in GoalFlow's Table 1.

Everything except the worker function is reused from the original script, so the
two stay in sync.

Usage mirrors the original, with the goal agent selected on the command line::

    python navsim/planning/script/run_pdm_score_recogdrive_goal.py \\
        agent=recogdrive_goal_agent \\
        agent.goal_mode=inpaint \\
        agent.checkpoint_path=/path/to/teacher.ckpt \\
        ...
"""

from typing import Any, Dict, List, Union
from pathlib import Path
from dataclasses import asdict
import traceback
import logging
import lzma
import pickle
import os
import uuid

import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache

import navsim.planning.script.run_pdm_score_recogdrive as base_script

logger = logging.getLogger(__name__)


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """Worker that evaluates each token with the ground-truth Scene attached."""
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting goal-conditioned worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    tokens_to_evaluate = sorted(tokens_to_evaluate)

    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        if dist.get_rank() == 0:
            logger.info(
                f"Rank {dist.get_rank()} processing scenario {idx+1} / {len(tokens_to_evaluate)} "
                f"in thread_id={thread_id}, node_id={node_id}"
            )

        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            # The only difference from run_pdm_score_recogdrive.py: the agent is
            # given the Scene, which is where the privileged goal point comes from.
            requires_scene = True
            agent_input = scene_loader.get_agent_input_from_token(token)
            if requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
            score_row["rank"] = dist.get_rank()
        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)

    return pickle.dumps(pdm_results)


# main() resolves run_pdm_score from the base module's globals at call time, so
# rebinding it here redirects the worker without duplicating the launcher.
base_script.run_pdm_score = run_pdm_score
main = base_script.main


if __name__ == "__main__":
    main()
