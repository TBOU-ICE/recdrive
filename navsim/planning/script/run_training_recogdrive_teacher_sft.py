"""Training entry for teacher-rollout SFT (Phase B). Additive file.

Train data comes from the four exclusive bucket indexes under
``il_training_newvlm_direct``: their union is the full navtrain + SimScale
trainable mixture, addressed by absolute path straight into the original /mnt
caches, so nothing is symlinked or copied.  Each sample is paired with the
teacher trajectory Phase A cached for it, and samples whose teacher missed the
logged GT by more than ``teacher_sft.max_teacher_fde_m`` are dropped.

Validation reuses ``navtrain_full_val_index.json``.  That index is in the
``direct-absolute-root-v1`` (cache_path + token -> relpath) shape, which is
exactly what ``CacheOnlyDataset``'s manifest loader already consumes, so
validation needs no new code and no rollout cache -- it scores against GT.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import os
import random
import signal

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.utils.rnn as rnn_utils
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module_teacher_sft import (
    AgentLightningTeacherSFT,
)
from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.planning.training.direct_index_dataset import (
    DirectIndexDataset,
    TeacherRolloutDataset,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

BUCKETS = (
    "general_or_no_tag",
    "progress_curbside_stopgo",
    "rule_intersection",
    "safety_dynamics_interaction",
)


class ResilientDataset(torch.utils.data.Dataset):
    """Skip cache shards that fail to decode or hang on the FUSE mount.

    Both failure modes are fatal under DDP: a raised error aborts the run, and a
    hung read leaves that rank out of the next NCCL collective so every other
    rank dies on the watchdog timeout.  Guard each load with a try/except AND a
    SIGALRM timeout (which fires in the DataLoader worker's main thread and can
    interrupt a blocked read), then resample a different index.
    """

    def __init__(self, base, max_retries: int = 8, load_timeout_s: int = 60):
        super().__init__()
        self.base = base
        self.max_retries = int(max_retries)
        self.load_timeout_s = int(load_timeout_s)

    def __len__(self) -> int:
        return len(self.base)

    def _load_one(self, j: int):
        if self.load_timeout_s <= 0:
            return self.base[j]

        def _on_timeout(signum, frame):
            raise TimeoutError(f"cache load exceeded {self.load_timeout_s}s")

        try:
            previous = signal.signal(signal.SIGALRM, _on_timeout)
        except (ValueError, OSError):
            return self.base[j]
        try:
            signal.alarm(self.load_timeout_s)
            return self.base[j]
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    def __getitem__(self, idx: int):
        n = len(self.base)
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            j = idx if attempt == 0 else random.randrange(n)
            try:
                return self._load_one(j)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Skipping unreadable sample idx=%d (attempt %d/%d): %r",
                    j, attempt + 1, self.max_retries, exc,
                )
        raise RuntimeError(
            f"Could not load a valid sample after {self.max_retries} retries"
        ) from last_exc


def collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Tuple[str, ...]]:
    features_list, targets_list, tokens_list = zip(*batch)

    features = {
        "history_trajectory": torch.stack(
            [f["history_trajectory"] for f in features_list], dim=0
        ).cpu(),
        "status_feature": torch.stack([f["status_feature"] for f in features_list], dim=0).cpu(),
        "last_hidden_state": rnn_utils.pad_sequence(
            [f["last_hidden_state"] for f in features_list],
            batch_first=True,
            padding_value=0.0,
        ).clone().detach(),
    }
    if "high_command_one_hot" in features_list[0]:
        features["high_command_one_hot"] = torch.stack(
            [f["high_command_one_hot"] for f in features_list], dim=0
        ).cpu()

    targets = {"trajectory": torch.stack([t["trajectory"] for t in targets_list], dim=0).cpu()}
    if "teacher_trajectory" in targets_list[0]:
        targets["teacher_trajectory"] = torch.stack(
            [t["teacher_trajectory"] for t in targets_list], dim=0
        ).cpu()
    return features, targets, tokens_list


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    lightning_module = AgentLightningTeacherSFT(agent=agent)
    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()

    sft_cfg = cfg.get("teacher_sft", {})
    index_root = Path(str(sft_cfg.get("index_root")))
    index_name = str(sft_cfg.get("index_name", "train_index.json"))
    rollout_path = str(sft_cfg.get("rollout_path"))
    max_teacher_fde_m = float(sft_cfg.get("max_teacher_fde_m", 3.0))
    val_index_path = sft_cfg.get("val_index_path", None)
    val_cache_path = sft_cfg.get("val_cache_path", None)

    base_train = DirectIndexDataset(
        bucket_indexes=[(b, str(index_root / b / index_name)) for b in BUCKETS],
        feature_builders=feature_builders,
        target_builders=target_builders,
    )
    train_data = TeacherRolloutDataset(
        base=base_train,
        rollout_path=rollout_path,
        max_teacher_fde_m=max_teacher_fde_m,
    )
    logger.info(
        "Teacher-rollout SFT train set: %d of %d indexed samples "
        "(%d without a rollout, %d dropped by teacher fde > %.2fm)",
        len(train_data), len(base_train), train_data.n_missing,
        train_data.n_filtered, max_teacher_fde_m,
    )

    # Validation scores the deployable student (predicted goal) against GT.
    val_data = CacheOnlyDataset(
        cache_path=str(val_cache_path),
        feature_builders=feature_builders,
        target_builders=target_builders,
        log_names=None,
        manifest_path=str(val_index_path),
    )
    logger.info("Validation set: %d samples", len(val_data))

    train_data = ResilientDataset(train_data)
    val_data = ResilientDataset(val_data)

    train_dataloader = DataLoader(
        train_data, collate_fn=collate_fn, shuffle=True, **cfg.dataloader.params
    )
    val_dataloader = DataLoader(
        val_data, collate_fn=collate_fn, shuffle=False, **cfg.dataloader.params
    )

    tensorboard_dir = cfg.get("tensorboard_dir", None)
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(Path(cfg.output_dir) / "checkpoints") if tensorboard_dir else None,
        monitor="val/loss_epoch",
        mode="min",
        save_top_k=5,
        every_n_epochs=1,
        save_last=True,
    )
    trainer_logger = (
        TensorBoardLogger(save_dir=str(tensorboard_dir), name=str(cfg.experiment_name))
        if tensorboard_dir
        else True
    )
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=[checkpoint_cb], logger=trainer_logger)

    ckpt_path = cfg.get("ckpt_path", None) or None
    if ckpt_path:
        logger.info("Resuming full trainer state from %s", ckpt_path)
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
