import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch

from navsim.planning.training.dataset import CacheOnlyDataset

logger = logging.getLogger(__name__)

EXCLUSIVE_BUCKET_TOKEN_FILES = {
    'safety_dynamics_interaction': 'exclusive_safety_dynamics_interaction_tokens.json',
    'rule_intersection': 'exclusive_rule_intersection_tokens.json',
    'progress_curbside_stopgo': 'exclusive_progress_curbside_stopgo_tokens.json',
    'general_or_no_tag': 'exclusive_general_or_no_tag_tokens.json',
}


def normalize_token(token: object) -> Optional[str]:
    if token is None:
        return None
    if isinstance(token, (bytes, bytearray)):
        token = token.hex()
    if not isinstance(token, str):
        return None
    token = token.strip().lower().replace('-', '')
    return token or None


def load_token_list(json_path: str) -> List[str]:
    path = Path(json_path)
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f'Expected token list in {path}, got {type(data).__name__}')

    tokens: List[str] = []
    seen = set()
    for token in data:
        normalized = normalize_token(token)
        if normalized and normalized not in seen:
            seen.add(normalized)
            tokens.append(normalized)

    if not tokens:
        raise ValueError(f'No valid tokens found in {path}')
    return tokens


def load_token_to_log_mapping(json_path: str) -> Dict[str, str]:
    path = Path(json_path)
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f'Expected token metadata dict in {path}, got {type(data).__name__}')

    token_to_log: Dict[str, str] = {}
    for token, metadata in data.items():
        normalized = normalize_token(token)
        if not normalized or not isinstance(metadata, dict):
            continue
        log_name = metadata.get('log_name')
        if isinstance(log_name, str) and log_name:
            token_to_log[normalized] = log_name

    if not token_to_log:
        raise ValueError(f'No token->log mappings found in {path}')
    return token_to_log


def load_complement_bucket_tokens(bucket_name: str, navtrain_output_dir: str) -> List[str]:
    if bucket_name not in EXCLUSIVE_BUCKET_TOKEN_FILES:
        raise ValueError(f'Unknown bucket name: {bucket_name}')

    tokens: List[str] = []
    seen = set()
    output_dir = Path(navtrain_output_dir)
    for name, filename in EXCLUSIVE_BUCKET_TOKEN_FILES.items():
        if name == bucket_name:
            continue
        for token in load_token_list(str(output_dir / filename)):
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens


def load_all_bucket_tokens(navtrain_output_dir: str) -> List[str]:
    tokens: List[str] = []
    seen = set()
    output_dir = Path(navtrain_output_dir)
    for filename in EXCLUSIVE_BUCKET_TOKEN_FILES.values():
        for token in load_token_list(str(output_dir / filename)):
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens


class TokenFilteredCacheOnlyDataset(CacheOnlyDataset):
    """Cache-only dataset with optional token whitelist filtering."""

    def __init__(
        self,
        cache_path: str,
        feature_builders,
        target_builders,
        log_names: Optional[List[str]] = None,
        tokens: Optional[Sequence[str]] = None,
        token_to_log: Optional[Dict[str, str]] = None,
    ):
        self._cache_path = Path(cache_path)
        if not self._cache_path.is_dir():
            raise AssertionError(f'Cache path {cache_path} does not exist!')

        if log_names is not None:
            self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
        else:
            self.log_names = [log_name for log_name in self._cache_path.iterdir()]

        self._feature_builders = feature_builders
        self._target_builders = target_builders

        if tokens is not None and token_to_log is not None:
            allowed_logs = {str(log_name) for log_name in self.log_names}
            self._valid_cache_paths = self._load_valid_caches_from_index(
                cache_path=self._cache_path,
                feature_builders=self._feature_builders,
                target_builders=self._target_builders,
                tokens=tokens,
                token_to_log=token_to_log,
                allowed_logs=allowed_logs,
            )
        else:
            token_filter = set(tokens) if tokens is not None else None
            self._valid_cache_paths = self._load_valid_caches(
                cache_path=self._cache_path,
                feature_builders=self._feature_builders,
                target_builders=self._target_builders,
                log_names=self.log_names,
                token_filter=token_filter,
            )
        self.tokens = list(self._valid_cache_paths.keys())

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,
        feature_builders,
        target_builders,
        log_names: List[Path],
        token_filter: Optional[set] = None,
    ) -> Dict[str, Path]:
        valid_cache_paths: Dict[str, Path] = {}

        for log_name in log_names:
            log_path = cache_path / log_name
            if not log_path.is_dir():
                continue
            for token_path in log_path.iterdir():
                token = normalize_token(token_path.name)
                if token_filter is not None and token not in token_filter:
                    continue

                found_caches: List[bool] = []
                for builder in feature_builders + target_builders:
                    data_dict_path = token_path / (builder.get_unique_name() + '.gz')
                    found_caches.append(data_dict_path.is_file())
                if all(found_caches) and token is not None:
                    valid_cache_paths[token] = token_path

        return valid_cache_paths

    @staticmethod
    def _load_valid_caches_from_index(
        cache_path: Path,
        feature_builders,
        target_builders,
        tokens: Sequence[str],
        token_to_log: Dict[str, str],
        allowed_logs: set,
    ) -> Dict[str, Path]:
        valid_cache_paths: Dict[str, Path] = {}
        total = len(tokens)

        for index, token in enumerate(tokens):
            if index > 0 and index % 10000 == 0:
                logger.info(
                    'Resolved cached tokens: %d/%d (found %d)',
                    index,
                    total,
                    len(valid_cache_paths),
                )

            normalized = normalize_token(token)
            if not normalized:
                continue

            log_name = token_to_log.get(normalized)
            if not log_name or log_name not in allowed_logs:
                continue

            token_path = cache_path / log_name / normalized
            if not token_path.is_dir():
                continue

            if all(
                (token_path / (builder.get_unique_name() + '.gz')).is_file()
                for builder in feature_builders + target_builders
            ):
                valid_cache_paths[normalized] = token_path

        logger.info('Resolved cached tokens: %d/%d (found %d)', total, total, len(valid_cache_paths))
        return valid_cache_paths


class RatioMixedCacheDataset(torch.utils.data.Dataset):
    """Fixed-size mixed dataset with epoch-wise resampling."""

    def __init__(
        self,
        full_dataset: torch.utils.data.Dataset,
        bucket_dataset: torch.utils.data.Dataset,
        full_ratio: float,
        bucket_ratio: float,
        epoch_size: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__()
        if full_ratio < 0 or bucket_ratio < 0:
            raise ValueError('Mix ratios must be non-negative')
        ratio_sum = full_ratio + bucket_ratio
        if ratio_sum <= 0:
            raise ValueError('At least one mix ratio must be positive')

        self.full_dataset = full_dataset
        self.bucket_dataset = bucket_dataset
        self.full_ratio = full_ratio / ratio_sum
        self.bucket_ratio = bucket_ratio / ratio_sum
        self.seed = seed
        self.epoch = 0

        if len(self.full_dataset) == 0:
            raise ValueError('Full dataset is empty')
        if len(self.bucket_dataset) == 0:
            raise ValueError('Bucket dataset is empty')

        self.epoch_size = epoch_size or len(self.full_dataset)
        if self.epoch_size < 2:
            raise ValueError('epoch_size must be at least 2')
        self._indices: List[Tuple[str, int]] = []
        self._rebuild_indices()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        rng = random.Random(self.seed + self.epoch)

        bucket_count = int(round(self.epoch_size * self.bucket_ratio))
        bucket_count = min(max(bucket_count, 1), self.epoch_size - 1)
        full_count = self.epoch_size - bucket_count

        indices: List[Tuple[str, int]] = []
        indices.extend(('full', rng.randrange(len(self.full_dataset))) for _ in range(full_count))
        indices.extend(('bucket', rng.randrange(len(self.bucket_dataset))) for _ in range(bucket_count))
        rng.shuffle(indices)
        self._indices = indices

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int):
        source, source_idx = self._indices[idx]
        if source == 'bucket':
            return self.bucket_dataset[source_idx]
        return self.full_dataset[source_idx]


class DatasetEpochCallback(pl.Callback):
    """Refreshes mixed dataset sampling at the start of each epoch."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        dataset = getattr(trainer.train_dataloader, 'dataset', None)
        if dataset is not None and hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(trainer.current_epoch)


def split_tokens_by_logs(
    tokens: Iterable[str],
    cache_path: str,
    log_names: Sequence[str],
    token_to_log: Optional[Dict[str, str]] = None,
) -> List[str]:
    log_name_set = {str(log_name) for log_name in log_names}
    cache_root = Path(cache_path)
    filtered: List[str] = []

    if token_to_log is not None:
        for token in tokens:
            normalized = normalize_token(token)
            if not normalized:
                continue
            log_name = token_to_log.get(normalized)
            if log_name in log_name_set and (cache_root / log_name / normalized).is_dir():
                filtered.append(normalized)
        return filtered

    for token in tokens:
        normalized = normalize_token(token)
        if not normalized:
            continue
        matched = False
        for log_name in log_name_set:
            token_dir = cache_root / log_name / normalized
            if token_dir.is_dir():
                matched = True
                break
        if matched:
            filtered.append(normalized)

    return filtered


def log_dataset_summary(
    bucket_name: str,
    full_train_size: int,
    bucket_train_size: int,
    val_size: int,
    full_ratio: float,
    bucket_ratio: float,
    epoch_size: int,
) -> None:
    logger.info('Bucket expert: %s', bucket_name)
    logger.info('Train full dataset size: %d', full_train_size)
    logger.info('Train bucket dataset size: %d', bucket_train_size)
    logger.info('Validation bucket dataset size: %d', val_size)
    logger.info('Train mix ratio full/bucket: %.3f / %.3f', full_ratio, bucket_ratio)
    logger.info('Train mixed epoch size: %d', epoch_size)
