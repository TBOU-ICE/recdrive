"""PDM scoring for the GoalBridge student (predicted goal, optional goal-off ablation).

Reuses ``run_pdm_score_recogdrive.py`` unchanged: no Scene is loaded, so the
privileged endpoint never reaches the agent.  The predicted-goal path lives in
``ReCogDriveGoalBridgeEvalAgent``.

This file exists so GoalBridge student eval cannot be confused with:

* ``agent=recogdrive_agent`` (plain DiT, drops the goal head)
* ``run_pdm_score_recogdrive_goal.py`` (privileged GT-goal teacher)

Usage::

    python navsim/planning/script/run_pdm_score_recogdrive_goalbridge.py \\
        agent=recogdrive_agent_goalbridge_eval \\
        agent.checkpoint_path=/path/to/goalbridge_student.ckpt \\
        ...
"""

import logging
import os
import random

import numpy as np
import torch

import navsim.planning.script.run_pdm_score_recogdrive as base_script

logger = logging.getLogger(__name__)

_ALLOWED_TARGETS = (
    "navsim.agents.recogdrive.recogdrive_goalbridge_eval_agent.ReCogDriveGoalBridgeEvalAgent",
)

_orig_run_pdm_score = base_script.run_pdm_score


def run_pdm_score(args):
    cfg = args[0]["cfg"]
    agent_target = str(getattr(cfg.agent, "_target_", "") or "")
    if agent_target not in _ALLOWED_TARGETS:
        raise RuntimeError(
            "run_pdm_score_recogdrive_goalbridge.py is for the GoalBridge student "
            f"only. Got agent._target_={agent_target!r}. "
            "Use agent=recogdrive_agent_goalbridge_eval. "
            "Plain DiT eval stays on run_pdm_score_recogdrive.py; "
            "privileged teacher eval stays on run_pdm_score_recogdrive_goal.py."
        )
    # The sampler starts from torch.randn. Fixing the per-rank RNG makes repeated
    # runs reproducible and, in particular, makes pred/off a controlled ablation.
    rank = int(os.environ.get("RANK", "0"))
    seed = int(getattr(cfg, "seed", 0)) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    goal_source = str(getattr(cfg.agent, "goal_source", "pred"))
    logger.info(
        "Starting GoalBridge student PDMS: goal_source=%s, no GT goal, seed=%d.",
        goal_source,
        seed,
    )
    return _orig_run_pdm_score(args)


base_script.run_pdm_score = run_pdm_score
main = base_script.main


if __name__ == "__main__":
    main()
