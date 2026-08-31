"""Stage-3 privileged goal-adapter training (cache-only, scene routed per job)."""
from __future__ import annotations

import json
import logging
import os
import random
import signal
from pathlib import Path
from typing import Optional, Set

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset

logger = logging.getLogger(__name__)
CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def norm_token(x):
    if isinstance(x, (bytes, bytearray)):
        x = x.hex()
    if not isinstance(x, str):
        return None
    x = x.strip().lower()
    base, sep, suffix = x.rpartition("-")
    if not (sep and suffix.isdigit() and len(suffix) == 3 and base):
        x = x.replace("-", "")
    return x or None


def load_tokens(path: str) -> Set[str]:
    raw = json.load(open(path, "r", encoding="utf-8"))
    seq = raw.keys() if isinstance(raw, dict) else raw
    out = {norm_token(x) for x in seq}
    out.discard(None)
    return out


def filter_cache(ds: CacheOnlyDataset, allowed: Set[str]):
    kept = {t: p for t, p in ds._valid_cache_paths.items() if norm_token(t) in allowed}
    ds._valid_cache_paths = kept
    ds.tokens = list(kept.keys())
    return ds


class ResilientCacheDataset(Dataset):
    """Skip corrupt or stalled cache shards instead of aborting every DDP rank."""

    def __init__(self, base: Dataset, max_retries: int = 8, load_timeout_s: int = 60):
        self.base = base
        self.max_retries = int(max_retries)
        self.load_timeout_s = int(load_timeout_s)

    def __len__(self):
        return len(self.base)

    def _load_one(self, idx: int):
        if self.load_timeout_s <= 0:
            return self.base[idx]

        def on_timeout(signum, frame):
            raise TimeoutError(f"cache load exceeded {self.load_timeout_s}s")

        try:
            previous = signal.signal(signal.SIGALRM, on_timeout)
        except (ValueError, OSError):
            return self.base[idx]
        try:
            signal.alarm(self.load_timeout_s)
            return self.base[idx]
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    def __getitem__(self, idx):
        size = len(self.base)
        last_error = None
        for attempt in range(self.max_retries):
            candidate = idx if attempt == 0 else random.randrange(size)
            try:
                return self._load_one(candidate)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Skipping unreadable cache sample idx=%d (attempt %d/%d): %r",
                    candidate,
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
        raise RuntimeError(
            f"Could not load a valid cache sample after {self.max_retries} retries"
        ) from last_error


def load_bad_token_dirs(path: str) -> Set[str]:
    """Load bad token directories with both /workspace and /mnt mount aliases."""
    aliases = (
        ("/workspace/datasets/", "/mnt/datasets/"),
        ("/mnt/datasets/", "/workspace/datasets/"),
    )
    bad_dirs: Set[str] = set()
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            shard = line.split("\t", 1)[0].strip()
            if not shard:
                continue
            token_dir = os.path.dirname(shard)
            bad_dirs.add(token_dir)
            for source, target in aliases:
                if token_dir.startswith(source):
                    bad_dirs.add(target + token_dir[len(source):])
    return bad_dirs


def prune_bad_tokens(ds: CacheOnlyDataset, bad_dirs: Set[str]) -> int:
    drop = [token for token, path in ds._valid_cache_paths.items() if str(path) in bad_dirs]
    for token in drop:
        ds._valid_cache_paths.pop(token, None)
    ds.tokens = list(ds._valid_cache_paths.keys())
    return len(drop)


class FixedRatioNavSim(Dataset):
    """Use every NAV bucket sample once and cycle SimScale to a target ratio."""
    def __init__(self, nav: Dataset, sim: Optional[Dataset], sim_ratio: float):
        self.nav = nav
        self.sim = sim
        self.sim_ratio = float(sim_ratio)
        if not 0 <= self.sim_ratio < 1:
            raise ValueError("sim_ratio must be in [0,1)")
        self.n_nav = len(nav)
        self.n_sim = 0 if sim is None else len(sim)
        self.sim_draws = 0
        if self.n_sim > 0 and self.sim_ratio > 0:
            self.sim_draws = int(round(self.n_nav * self.sim_ratio / max(1e-6, 1 - self.sim_ratio)))

    def __len__(self):
        return self.n_nav + self.sim_draws

    def __getitem__(self, idx):
        if idx < self.n_nav:
            return self.nav[idx]
        if self.n_sim == 0:
            return self.nav[idx % self.n_nav]
        # A multiplicative permutation avoids repeatedly pairing the same prefix.
        j = ((idx - self.n_nav) * 104729) % self.n_sim
        return self.sim[j]


def collate(batch):
    features_list, targets_list, tokens_list = zip(*batch)
    return (
        {
            "history_trajectory": torch.stack([x["history_trajectory"] for x in features_list]).cpu(),
            "high_command_one_hot": torch.stack([x["high_command_one_hot"] for x in features_list]).cpu(),
            "status_feature": torch.stack([x["status_feature"] for x in features_list]).cpu(),
            "last_hidden_state": rnn_utils.pad_sequence(
                [x["last_hidden_state"] for x in features_list], batch_first=True, padding_value=0.0
            ).clone().detach(),
        },
        {"trajectory": torch.stack([x["trajectory"] for x in targets_list]).cpu()},
        tokens_list,
    )


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    local_rank = int(os.getenv("LOCAL_RANK", 0)); rank = int(os.getenv("RANK", 0)); world = int(os.getenv("WORLD_SIZE", 1))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)

    agent = instantiate(cfg.agent)
    agent.initialize()
    lightning = AgentLightningModule(agent=agent)
    fbs, tbs = agent.get_feature_builders(), agent.get_target_builders()

    nav_tokens = str(cfg.priv_goal_nav_bucket_tokens)
    nav_manifest = cfg.get("priv_goal_nav_manifest", None)
    nav_train = CacheOnlyDataset(cfg.cache_path, fbs, tbs, cfg.train_logs, manifest_path=nav_manifest)
    nav_val = CacheOnlyDataset(cfg.cache_path, fbs, tbs, cfg.val_logs, manifest_path=nav_manifest)
    allowed_nav = load_tokens(nav_tokens)
    filter_cache(nav_train, allowed_nav); filter_cache(nav_val, allowed_nav)
    if len(nav_train) == 0:
        raise RuntimeError("NAV bucket contains zero cached training samples")

    bad_list_path = os.environ.get("SCENE_ROUTER_BAD_CACHE_LIST", "").strip()
    bad_dirs: Set[str] = set()
    if bad_list_path and os.path.isfile(bad_list_path):
        bad_dirs = load_bad_token_dirs(bad_list_path)
        dropped_train = prune_bad_tokens(nav_train, bad_dirs)
        dropped_val = prune_bad_tokens(nav_val, bad_dirs)
        logger.info(
            "Pruned known-bad NAV shards from %s: %d train + %d val",
            bad_list_path,
            dropped_train,
            dropped_val,
        )
    elif bad_list_path:
        logger.warning("SCENE_ROUTER_BAD_CACHE_LIST set but not found: %s", bad_list_path)

    sim_sets = []
    sim_paths = list(cfg.get("priv_goal_sim_cache_paths", []) or [])
    sim_manifests = list(cfg.get("priv_goal_sim_manifests", []) or [])
    sim_token_jsons = list(cfg.get("priv_goal_sim_bucket_tokens", []) or [])
    for i, path in enumerate(sim_paths):
        manifest = sim_manifests[i] if i < len(sim_manifests) else None
        tok_json = sim_token_jsons[i] if i < len(sim_token_jsons) else None
        ds = CacheOnlyDataset(path, fbs, tbs, None, manifest_path=manifest)
        if tok_json:
            filter_cache(ds, load_tokens(tok_json))
        if bad_dirs:
            dropped = prune_bad_tokens(ds, bad_dirs)
            if dropped:
                logger.info("Pruned %d known-bad SimScale shards from %s", dropped, path)
        if len(ds):
            sim_sets.append(ds)
            logger.info("SimScale bucket cache %s -> %d samples", path, len(ds))
    sim = None if not sim_sets else (sim_sets[0] if len(sim_sets) == 1 else ConcatDataset(sim_sets))
    sim_ratio = float(cfg.get("priv_goal_sim_ratio", 0.40))
    train = FixedRatioNavSim(nav_train, sim, sim_ratio)
    logger.info("Privileged stage3: nav=%d sim_pool=%d effective_total=%d target_sim_ratio=%.3f",
                len(nav_train), 0 if sim is None else len(sim), len(train), sim_ratio)

    max_retries = int(os.environ.get("CACHE_LOAD_MAX_RETRIES", "8"))
    load_timeout_s = int(os.environ.get("CACHE_LOAD_TIMEOUT_SEC", "60"))
    train = ResilientCacheDataset(train, max_retries=max_retries, load_timeout_s=load_timeout_s)
    nav_val = ResilientCacheDataset(nav_val, max_retries=max_retries, load_timeout_s=load_timeout_s)
    train_loader = DataLoader(train, shuffle=True, collate_fn=collate, **cfg.dataloader.params)
    val_loader = DataLoader(nav_val, shuffle=False, collate_fn=collate, **cfg.dataloader.params)
    cb = pl.callbacks.ModelCheckpoint(monitor="val/loss_epoch", mode="min", save_top_k=5, save_last=True, every_n_epochs=1)
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=[cb])
    trainer.fit(lightning, train_loader, val_loader, ckpt_path=(cfg.get("ckpt_path", None) or None))


if __name__ == "__main__":
    main()
