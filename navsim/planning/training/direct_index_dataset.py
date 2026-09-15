"""Cache dataset driven by the bucket-expert *direct index* files. Additive file.

The IL bucket-expert runs (``run_recogdrive_bucket_expert_il_goal_newvlm.sh``)
read their data through per-bucket index JSONs produced by
``scripts/data/prep_bucket_il_direct_indexes.py``:

    {"samples": [{"source": ..., "token": ..., "path": "<absolute token dir>"}, ...]}

Each entry points straight at a token directory in the original cache, so the
navtrain and simscale rounds are already merged and no manifest, token
whitelist, or link farm is involved.  Reading the four bucket indexes therefore
yields both the full training set and the scenario routing for free -- the
bucket is simply which index file a sample came from.

This reads the same files from the OPD repo, whose ``CacheOnlyDataset`` has no
``index_path`` support.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.planning.training.dataset import load_feature_target_from_pickle

logger = logging.getLogger(__name__)


class DirectIndexCacheDataset(torch.utils.data.Dataset):
    """Loads cached features/targets from one or more direct index JSONs.

    :param index_paths: index JSONs to read, in order.
    :param labels: optional per-index label (the bucket name) exposed through
        ``bucket_for_token``; ``None`` disables labelling.
    :param feature_builders: feature builders whose ``get_unique_name()`` names
        the ``.gz`` to load from each token directory.
    :param target_builders: same, for targets.
    """

    def __init__(
        self,
        index_paths: Sequence[str],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        labels: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        if not index_paths:
            raise ValueError("At least one direct index path is required.")
        if labels is not None and len(labels) != len(index_paths):
            raise ValueError(f"labels ({len(labels)}) must match index_paths ({len(index_paths)})")

        self._feature_builders = feature_builders
        self._target_builders = target_builders

        self.tokens: List[str] = []
        self.paths: List[Path] = []
        self.labels: List[Optional[str]] = []
        seen: Dict[str, int] = {}
        per_index: Dict[str, int] = {}
        duplicates = 0

        for i, index_path in enumerate(index_paths):
            path = Path(index_path)
            if not path.is_file():
                raise FileNotFoundError(f"direct index not found: {index_path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            samples = payload.get("samples", [])
            if not samples:
                raise ValueError(f"direct index is empty: {index_path}")
            label = labels[i] if labels is not None else None
            kept = 0
            for sample in samples:
                token = str(sample["token"])
                if token in seen:
                    duplicates += 1
                    continue
                seen[token] = len(self.tokens)
                self.tokens.append(token)
                self.paths.append(Path(sample["path"]))
                self.labels.append(label)
                kept += 1
            per_index[label or path.name] = kept

        self._token_to_row = seen
        logger.info(
            "DirectIndexCacheDataset: %d samples from %d index file(s) %s (%d duplicate tokens skipped)",
            len(self.tokens), len(index_paths), per_index, duplicates,
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def bucket_for_token(self, token: str) -> Optional[str]:
        row = self._token_to_row.get(token)
        return self.labels[row] if row is not None else None

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        token_path = self.paths[idx]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            features.update(load_feature_target_from_pickle(token_path / (builder.get_unique_name() + ".gz")))

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            targets.update(load_feature_target_from_pickle(token_path / (builder.get_unique_name() + ".gz")))

        return features, targets, self.tokens[idx]


def prune_bad_tokens(dataset: DirectIndexCacheDataset, bad_dirs) -> int:
    """Drop tokens whose cache directory is on the known-bad-shard list."""
    if not bad_dirs:
        return 0
    keep = [i for i, p in enumerate(dataset.paths) if str(p) not in bad_dirs]
    dropped = len(dataset.tokens) - len(keep)
    if dropped:
        dataset.tokens = [dataset.tokens[i] for i in keep]
        dataset.paths = [dataset.paths[i] for i in keep]
        dataset.labels = [dataset.labels[i] for i in keep]
        dataset._token_to_row = {t: i for i, t in enumerate(dataset.tokens)}
    return dropped
