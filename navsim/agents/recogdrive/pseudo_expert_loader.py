"""
Pseudo-expert trajectory lookup table.

Reads dataset_decoupled_v2_clean.pkl (Stage-1 pseudo-expert trajectory package
from Clover) and builds a {token -> best_traj} dict where best_traj is the
trajectory with the highest pdm_score among all scored candidates.

Data layout per item in the pkl:
  token                  : str   scene token matching NAVSIM cache keys
  valid                  : bool  whether the item passed pre-checks
  trajectories_relative  : list[(8,3) ndarray]  ego-frame (x[m], y[m], heading[rad])
  scores                 : list[dict]  each dict has 'pdm_score' in [0,1] + sub-scores
  (other fields not used here: trajectories_global, generation, precheck, stats)
"""
import logging
import pickle
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def build_expert_lookup(pkl_path: str) -> Dict[str, np.ndarray]:
    """
    Load pseudo-expert data and return {token: best_traj}.

    best_traj shape: (8, 3) float32, ego-relative [x(m), y(m), heading(rad)].
    Tokens that have no valid scores, mismatched list lengths, or valid=False
    are omitted from the dict — callers should handle missing keys gracefully.
    """
    logger.info("Loading pseudo-expert lookup from %s", pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    lookup: Dict[str, np.ndarray] = {}
    n_skipped = 0

    for item in data:
        # Skip items flagged as invalid or missing scores
        if not item.get("valid", False):
            n_skipped += 1
            continue
        scores = item.get("scores", [])
        trajs  = item.get("trajectories_relative", [])
        if not scores or not trajs or len(scores) != len(trajs):
            n_skipped += 1
            continue

        # Pick the trajectory index with the highest pdm_score
        best_idx = max(range(len(scores)), key=lambda i: scores[i]["pdm_score"])

        # Only include tokens where at least one trajectory has PDMS > 0
        if scores[best_idx]["pdm_score"] <= 0.0:
            n_skipped += 1
            continue

        lookup[item["token"]] = np.array(trajs[best_idx], dtype=np.float32)  # (8, 3)

    logger.info(
        "Pseudo-expert lookup built: %d tokens, %d skipped (invalid / no score).",
        len(lookup),
        n_skipped,
    )
    return lookup
