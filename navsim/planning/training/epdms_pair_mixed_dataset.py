"""Pair-aware mixed cache dataset for EPDMS RL expert training.

Three unit sources per batch:

* ``navtrain_full``   -- adjacent-frame PAIR units from the full navtrain cache
                         (both frames trainable, EC computable);
* ``navtrain_bucket`` -- PAIR units whose *current* frame belongs to the
                         expert's scene bucket (over-sampling of the bucket);
* ``simscale_bucket`` -- SINGLE units from the SimScale synthetic caches
                         (no adjacent frames exist there, so no EC term).

The batch is flattened as ``[prev_0, cur_0, prev_1, cur_1, ..., single_0, ...]``
and pair metadata travels inside the feature dict (``epdms_num_pairs``,
``epdms_pair_dt``), so the stock ``AgentLightningDiT`` step signature
``(features, targets, tokens)`` keeps working unchanged.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.utils.rnn as rnn_utils

from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.planning.training.dataset import CacheOnlyDataset, load_feature_target_from_pickle

logger = logging.getLogger(__name__)


def normalize_token(token: str) -> str:
    """Lowercase; keep SimScale's ``-NNN`` variant suffix, strip other dashes."""
    token = str(token).strip().lower()
    base, sep, suffix = token.rpartition("-")
    if sep and suffix.isdigit() and len(suffix) == 3 and base:
        return token
    return token.replace("-", "")


def _scan_cache(
    cache_path: Path,
    feature_builders: Sequence[AbstractFeatureBuilder],
    target_builders: Sequence[AbstractTargetBuilder],
    log_names: Optional[Sequence[str]] = None,
    manifest_path: Optional[str] = None,
) -> Dict[str, Path]:
    """token -> token_dir for one agent-feature cache root.

    Walking ~100k token dirs on CPFS takes 30-40 min, so a prebuilt manifest
    (see scripts/data/build_agent_cache_manifests.py) is strongly preferred;
    the walk is only a fallback.
    """
    builder_names = sorted(b.get_unique_name() for b in list(feature_builders) + list(target_builders))
    if manifest_path and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get("cache_path") != str(cache_path):
            raise ValueError(f"manifest {manifest_path} was built for {manifest.get('cache_path')}, not {cache_path}")
        if sorted(manifest.get("builders", [])) != builder_names:
            raise ValueError(f"manifest {manifest_path} builders mismatch: {manifest.get('builders')} vs {builder_names}")
        wanted = set(log_names) if log_names is not None else None
        out: Dict[str, Path] = {}
        for token, rel in manifest["tokens"].items():
            log_name = rel.split("/", 1)[0]
            if wanted is not None and log_name not in wanted:
                continue
            out[normalize_token(token)] = cache_path / rel
        logger.info("loaded cache manifest %s: %d tokens", manifest_path, len(out))
        return out

    logger.warning("no manifest for %s -- falling back to slow directory walk", cache_path)
    if log_names is not None:
        wanted = set(log_names)
        logs = [p for p in cache_path.iterdir() if p.name in wanted]
    else:
        logs = [p for p in cache_path.iterdir() if p.is_dir()]
    valid = CacheOnlyDataset._load_valid_caches(
        cache_path=cache_path,
        feature_builders=list(feature_builders),
        target_builders=list(target_builders),
        log_names=logs,
    )
    return {normalize_token(k): v for k, v in valid.items()}


class EpdmsPairMixedDataset(torch.utils.data.Dataset):
    """Unit-level dataset: item = one pair unit or one single unit."""

    def __init__(
        self,
        nav_cache_path: str,
        pair_table_path: str,
        bucket_tokens_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        sim_cache_paths: Optional[List[str]] = None,
        sim_token_list_paths: Optional[List[str]] = None,
        train_log_names: Optional[List[str]] = None,
        nav_allowed_tokens: Optional[set] = None,
        sim_allowed_tokens: Optional[set] = None,
        nav_manifest: Optional[str] = None,
        sim_manifests: Optional[List[str]] = None,
    ):
        super().__init__()
        self._feature_builders = feature_builders
        self._target_builders = target_builders

        nav_paths = _scan_cache(
            Path(nav_cache_path), feature_builders, target_builders, train_log_names, manifest_path=nav_manifest
        )
        logger.info("navtrain cache: %d tokens", len(nav_paths))
        if nav_allowed_tokens is not None:
            # reward-side availability (v2 metric cache); tolerate partial builds
            nav_allowed = {normalize_token(t) for t in nav_allowed_tokens}
            before = len(nav_paths)
            nav_paths = {t: p for t, p in nav_paths.items() if t in nav_allowed}
            logger.info("navtrain tokens with v2 metric cache: %d/%d", len(nav_paths), before)

        pair_table = json.loads(Path(pair_table_path).read_text())
        pairs = [
            (normalize_token(prev), normalize_token(cur), float(dt))
            for prev, cur, dt in pair_table["pairs"]
            if normalize_token(prev) in nav_paths and normalize_token(cur) in nav_paths
        ]
        logger.info("usable adjacent pairs: %d", len(pairs))

        bucket_tokens = {normalize_token(t) for t in json.loads(Path(bucket_tokens_path).read_text())}
        logger.info("bucket tokens (raw list): %d", len(bucket_tokens))

        self.units: List[Dict[str, Any]] = []
        self.unit_source: List[str] = []
        for prev, cur, dt in pairs:
            self.units.append(
                {"kind": "pair", "prev": prev, "cur": cur, "dt": dt, "paths": (nav_paths[prev], nav_paths[cur])}
            )
            self.unit_source.append("navtrain_bucket" if cur in bucket_tokens else "navtrain_full")

        sim_cache_paths = sim_cache_paths or []
        sim_token_list_paths = sim_token_list_paths or []
        assert len(sim_cache_paths) == len(sim_token_list_paths), "sim cache dirs and token lists must align"
        sim_metric_allowed = (
            {normalize_token(t) for t in sim_allowed_tokens} if sim_allowed_tokens is not None else None
        )
        sim_manifests = sim_manifests or [None] * len(sim_cache_paths)
        assert len(sim_manifests) == len(sim_cache_paths), "sim_manifests must align with sim_cache_paths"
        for cache_dir, token_list_path, manifest in zip(sim_cache_paths, sim_token_list_paths, sim_manifests):
            sim_paths = _scan_cache(Path(cache_dir), feature_builders, target_builders, manifest_path=manifest)
            allowed = {normalize_token(t) for t in json.loads(Path(token_list_path).read_text())}
            if sim_metric_allowed is not None:
                allowed &= sim_metric_allowed
            kept = 0
            for token, path in sim_paths.items():
                if token in allowed:
                    self.units.append({"kind": "single", "token": token, "path": path})
                    self.unit_source.append("simscale_bucket")
                    kept += 1
            logger.info("simscale source %s: %d/%d tokens kept", cache_dir, kept, len(sim_paths))

        counts: Dict[str, int] = {}
        for src in self.unit_source:
            counts[src] = counts.get(src, 0) + 1
        self.source_counts = counts
        logger.info("unit counts per source: %s", counts)

    def __len__(self) -> int:
        return len(self.units)

    @property
    def unit_sizes(self) -> List[int]:
        return [2 if u["kind"] == "pair" else 1 for u in self.units]

    def _load_one(self, token_dir: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            features.update(load_feature_target_from_pickle(token_dir / (builder.get_unique_name() + ".gz")))
        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            targets.update(load_feature_target_from_pickle(token_dir / (builder.get_unique_name() + ".gz")))
        return features, targets

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        unit = self.units[idx]
        if unit["kind"] == "pair":
            prev_feat, prev_tgt = self._load_one(unit["paths"][0])
            cur_feat, cur_tgt = self._load_one(unit["paths"][1])
            return {
                "kind": "pair",
                "dt": unit["dt"],
                "samples": [
                    (prev_feat, prev_tgt, unit["prev"]),
                    (cur_feat, cur_tgt, unit["cur"]),
                ],
            }
        feat, tgt = self._load_one(unit["path"])
        return {"kind": "single", "samples": [(feat, tgt, unit["token"])]}


class EpdmsUnitBatchSampler(torch.utils.data.Sampler):
    """Weighted with-replacement batch sampler over units with a per-batch
    SAMPLE budget (a pair consumes 2 samples, a single consumes 1).

    Source ratios are enforced at the *sample* level: P(unit of source s)
    proportional to ratio_s / (count_s * size_s).
    """

    def __init__(
        self,
        dataset: EpdmsPairMixedDataset,
        source_ratios: Dict[str, float],
        samples_per_batch: int,
        steps_per_epoch: int,
        seed: int = 0,
        rank: int = 0,
    ):
        self.samples_per_batch = int(samples_per_batch)
        self.steps_per_epoch = int(steps_per_epoch)

        sizes = dataset.unit_sizes
        counts = dataset.source_counts
        missing = [s for s, r in source_ratios.items() if r > 0 and counts.get(s, 0) == 0]
        if missing:
            logger.warning("sources with ratio>0 but no units: %s (ratios renormalized)", missing)
        weights = torch.zeros(len(dataset), dtype=torch.double)
        for i, (src, size) in enumerate(zip(dataset.unit_source, sizes)):
            ratio = float(source_ratios.get(src, 0.0))
            if ratio > 0:
                weights[i] = ratio / (counts[src] * size)
        if float(weights.sum()) <= 0:
            raise ValueError(f"no sampleable units for ratios {source_ratios} on counts {counts}")
        self.weights = weights
        self.sizes = sizes
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed) * 100003 + int(rank))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        max_units = self.samples_per_batch  # upper bound (all singles)
        for _ in range(self.steps_per_epoch):
            drawn = torch.multinomial(self.weights, max_units, replacement=True, generator=self.generator).tolist()
            batch: List[int] = []
            n_samples = 0
            for idx in drawn:
                batch.append(idx)
                n_samples += self.sizes[idx]
                if n_samples >= self.samples_per_batch:
                    break
            yield batch


def epdms_pair_collate(
    units: List[Dict[str, Any]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Tuple[str, ...]]:
    """Flatten units to ``[prev_0, cur_0, ..., single_0, ...]`` and stack tensors."""
    ordered: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]] = []
    pair_dt: List[float] = []
    for unit in units:
        if unit["kind"] == "pair":
            ordered.extend(unit["samples"])
            pair_dt.append(unit["dt"])
    for unit in units:
        if unit["kind"] == "single":
            ordered.extend(unit["samples"])

    features_list, targets_list, tokens_list = zip(*ordered)
    features = {
        "history_trajectory": torch.stack([f["history_trajectory"] for f in features_list], dim=0).cpu(),
        "high_command_one_hot": torch.stack([f["high_command_one_hot"] for f in features_list], dim=0).cpu(),
        "status_feature": torch.stack([f["status_feature"] for f in features_list], dim=0).cpu(),
        "last_hidden_state": rnn_utils.pad_sequence(
            [f["last_hidden_state"] for f in features_list], batch_first=True, padding_value=0.0
        ).clone().detach(),
        "epdms_num_pairs": torch.tensor(len(pair_dt), dtype=torch.long),
        "epdms_pair_dt": torch.tensor(pair_dt, dtype=torch.float64),
    }
    targets = {"trajectory": torch.stack([t["trajectory"] for t in targets_list], dim=0).cpu()}
    return features, targets, tuple(tokens_list)


def plain_collate(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Tuple[str, ...]]:
    """Validation collate for CacheOnlyDataset triples (no pair metadata)."""
    features_list, targets_list, tokens_list = zip(*batch)
    features = {
        "history_trajectory": torch.stack([f["history_trajectory"] for f in features_list], dim=0).cpu(),
        "high_command_one_hot": torch.stack([f["high_command_one_hot"] for f in features_list], dim=0).cpu(),
        "status_feature": torch.stack([f["status_feature"] for f in features_list], dim=0).cpu(),
        "last_hidden_state": rnn_utils.pad_sequence(
            [f["last_hidden_state"] for f in features_list], batch_first=True, padding_value=0.0
        ).clone().detach(),
        "epdms_num_pairs": torch.tensor(0, dtype=torch.long),
        "epdms_pair_dt": torch.tensor([], dtype=torch.float64),
    }
    targets = {"trajectory": torch.stack([t["trajectory"] for t in targets_list], dim=0).cpu()}
    return features, targets, tuple(normalize_token(t) for t in tokens_list)
