"""Datasets backed by ``direct-absolute-paths-v1`` indexes. Additive file.

The bucket IL teachers are trained from prebuilt indexes under
``il_training_newvlm_direct/<bucket>/train_index.json``.  Each index stores an
absolute path per token, pointing straight at the original ``/mnt`` cache, so no
symlink farm (and no hard link, which could not cross the PolarFS / NAS mount
boundary anyway) is ever created.

Two things make these indexes the natural input for teacher-rollout SFT:

* the four buckets are *exclusive*, so their union is exactly the trainable
  navtrain + SimScale mixture, and
* membership in an index already encodes which privileged teacher owns a sample,
  which is the same routing ``token_to_bucket_json`` would provide.

``DirectIndexDataset`` yields the usual ``(features, targets, token)`` triple so
it drops into the existing collate functions.  ``TeacherRolloutDataset`` wraps it
and injects the cached teacher trajectory as an extra target.
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

INDEX_FORMAT = "direct-absolute-paths-v1"
ROLLOUT_FORMAT = "teacher-rollout-v1"


def load_direct_index(index_path: str) -> List[Dict[str, str]]:
    """Read one ``direct-absolute-paths-v1`` index into its raw sample list."""
    path = Path(index_path)
    if not path.is_file():
        raise FileNotFoundError(f"direct index not found: {index_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fmt = payload.get("format")
    if fmt != INDEX_FORMAT:
        raise ValueError(f"{index_path}: expected format {INDEX_FORMAT!r}, got {fmt!r}")
    samples = payload.get("samples", [])
    if not samples:
        raise ValueError(f"{index_path}: index contains no samples")
    return samples


class DirectIndexDataset(torch.utils.data.Dataset):
    """Cache dataset addressed by absolute per-token paths from bucket indexes.

    :param bucket_indexes: ordered ``(bucket_name, index_path)`` pairs.  A token
        that appears in more than one index keeps its first occurrence, which
        matters only if the indexes stop being exclusive.
    """

    def __init__(
        self,
        bucket_indexes: Sequence[Tuple[str, str]],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        allowed_tokens: Optional[set] = None,
    ):
        super().__init__()
        self._feature_builders = list(feature_builders)
        self._target_builders = list(target_builders)

        self.tokens: List[str] = []
        self.paths: List[Path] = []
        self.buckets: List[str] = []
        seen: set = set()
        per_bucket: Dict[str, int] = {}

        for bucket, index_path in bucket_indexes:
            kept = 0
            for sample in load_direct_index(index_path):
                token = sample["token"]
                if token in seen:
                    continue
                if allowed_tokens is not None and token not in allowed_tokens:
                    continue
                seen.add(token)
                self.tokens.append(token)
                self.paths.append(Path(sample["path"]))
                self.buckets.append(bucket)
                kept += 1
            per_bucket[bucket] = kept
            logger.info("direct index %s -> bucket %s: %d samples", index_path, bucket, kept)

        if not self.tokens:
            raise RuntimeError("DirectIndexDataset is empty after loading every index")
        self.per_bucket_counts = per_bucket
        logger.info("DirectIndexDataset: %d samples across %d buckets", len(self.tokens), len(per_bucket))

    def __len__(self) -> int:
        return len(self.tokens)

    def _load(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        token_path = self.paths[idx]
        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            features.update(
                load_feature_target_from_pickle(token_path / (builder.get_unique_name() + ".gz"))
            )
        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            targets.update(
                load_feature_target_from_pickle(token_path / (builder.get_unique_name() + ".gz"))
            )
        return features, targets

    def __getitem__(self, idx: int):
        features, targets = self._load(idx)
        return features, targets, self.tokens[idx]


def load_teacher_rollout(rollout_path: str) -> Dict[str, object]:
    """Load a ``teacher-rollout-v1`` payload produced by the generation script."""
    path = Path(rollout_path)
    if not path.is_file():
        raise FileNotFoundError(f"teacher rollout cache not found: {rollout_path}")
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    fmt = payload.get("format")
    if fmt != ROLLOUT_FORMAT:
        raise ValueError(f"{rollout_path}: expected format {ROLLOUT_FORMAT!r}, got {fmt!r}")
    return payload


class TeacherRolloutDataset(torch.utils.data.Dataset):
    """Direct-index cache samples paired with their cached teacher trajectory.

    Samples without a cached rollout, or whose teacher missed the logged GT
    trajectory by more than ``max_teacher_fde_m``, are dropped: the teacher is
    not an oracle, and regressing onto its failures teaches the student nothing
    that OPD would later reward.
    """

    def __init__(
        self,
        base: DirectIndexDataset,
        rollout_path: str,
        max_teacher_fde_m: float = 3.0,
    ):
        super().__init__()
        self.base = base

        payload = load_teacher_rollout(rollout_path)
        tokens: List[str] = payload["tokens"]
        traj: torch.Tensor = payload["trajectory"].float()
        fde: torch.Tensor = payload["fde_to_gt"].float()
        row_of = {token: i for i, token in enumerate(tokens)}

        self.indices: List[int] = []
        self.rows: List[int] = []
        n_missing = 0
        n_filtered = 0
        for i, token in enumerate(base.tokens):
            row = row_of.get(token)
            if row is None:
                n_missing += 1
                continue
            if max_teacher_fde_m > 0.0 and float(fde[row]) > max_teacher_fde_m:
                n_filtered += 1
                continue
            self.indices.append(i)
            self.rows.append(row)

        if not self.indices:
            raise RuntimeError(
                f"No usable teacher rollouts: {n_missing} missing, {n_filtered} above "
                f"max_teacher_fde_m={max_teacher_fde_m}"
            )
        self.trajectory = traj
        self.fde_to_gt = fde
        self.n_missing = n_missing
        self.n_filtered = n_filtered
        logger.info(
            "TeacherRolloutDataset: %d usable (%d missing rollout, %d dropped by fde > %.2fm)",
            len(self.indices), n_missing, n_filtered, max_teacher_fde_m,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        features, targets = self.base._load(base_idx)
        targets["teacher_trajectory"] = self.trajectory[self.rows[idx]].clone()
        return features, targets, self.base.tokens[base_idx]
