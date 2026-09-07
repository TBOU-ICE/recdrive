from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.planning.training.dataset import load_feature_target_from_pickle


class TemporalCachePairDataset(torch.utils.data.Dataset):
    """Cache-backed dataset returning adjacent NAVSIM token pairs.

    Each item is a pair of cached samples from the same log, ordered by timestamp:
        ((features_t, targets_t, token_t), (features_next, targets_next, token_next))

    Used only by run_training_recogdrive_temporal_multi_teacher_dit_distill.py.
    All original datasets are left untouched.
    """

    def __init__(
        self,
        scene_loader: SceneLoader,
        cache_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: Optional[List[str]] = None,
        min_dt_s: float = 0.1,
        max_dt_s: float = 1.1,
        max_pairs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._scene_loader = scene_loader
        self._cache_path = Path(cache_path)
        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._min_dt_us = int(float(min_dt_s) * 1e6)
        self._max_dt_us = int(float(max_dt_s) * 1e6)
        self._max_pairs = max_pairs if max_pairs and int(max_pairs) > 0 else None

        if not self._cache_path.is_dir():
            raise FileNotFoundError(f"Cache path does not exist: {self._cache_path}")

        self._valid_cache_paths = self._load_valid_caches(log_names=log_names)
        self.pairs = self._build_pairs_from_scene_loader()
        if self._max_pairs is not None:
            self.pairs = self.pairs[: self._max_pairs]

        if len(self.pairs) == 0:
            raise RuntimeError(
                "TemporalCachePairDataset found 0 adjacent token pairs. "
                "Check cache_path, train_logs/val_logs, and min_dt_s/max_dt_s."
            )

    def _load_valid_caches(self, log_names: Optional[List[str]]) -> Dict[str, Path]:
        valid: Dict[str, Path] = {}
        allowed_logs = set(str(x) for x in log_names) if log_names is not None else None
        log_dirs = [p for p in self._cache_path.iterdir() if p.is_dir()]
        for log_path in tqdm(log_dirs, desc="Loading temporal valid caches"):
            if allowed_logs is not None and log_path.name not in allowed_logs:
                continue
            for token_path in log_path.iterdir():
                if not token_path.is_dir():
                    continue
                found = []
                for builder in self._feature_builders + self._target_builders:
                    found.append((token_path / (builder.get_unique_name() + ".gz")).is_file())
                if all(found):
                    valid[token_path.name] = token_path
        return valid

    def _build_pairs_from_scene_loader(self) -> List[Tuple[str, str]]:
        groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

        for token, frame_list in self._scene_loader.scene_frames_dicts.items():
            if token not in self._valid_cache_paths:
                continue
            if not frame_list:
                continue
            current_idx = int(self._scene_loader._scene_filter.num_history_frames) - 1
            current_idx = max(0, min(current_idx, len(frame_list) - 1))
            frame = frame_list[current_idx]
            log_name = str(frame.get("log_name", frame_list[0].get("log_name", "")))
            timestamp = int(frame.get("timestamp", 0))
            groups[log_name].append((timestamp, token))

        pairs: List[Tuple[str, str]] = []
        for _, items in groups.items():
            items = sorted(items, key=lambda x: x[0])
            for (ts_a, tok_a), (ts_b, tok_b) in zip(items[:-1], items[1:]):
                dt = ts_b - ts_a
                if self._min_dt_us <= dt <= self._max_dt_us:
                    pairs.append((tok_a, tok_b))
        return pairs

    def _load_token(self, token: str):
        token_path = self._valid_cache_paths[token]
        features = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            features.update(load_feature_target_from_pickle(data_dict_path))
        targets = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            targets.update(load_feature_target_from_pickle(data_dict_path))
        return features, targets, token

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        token_t, token_next = self.pairs[idx]
        return self._load_token(token_t), self._load_token(token_next)
