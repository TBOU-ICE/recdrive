"""Add-only deterministic data/probe utilities for Phase-C SFT -> OPD.

Nothing in this module is imported by existing training entrypoints.  The new
``run_training_recogdrive_student_to_teacher_sft_opd.py`` opts into it.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers.feature_extraction_utils import BatchFeature

from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.planning.training.dataset import load_feature_target_from_pickle


PROBE_FORMAT = "student-to-teacher-opd-probe-v1"
PROBE_METRICS = (
    "student_teacher_ade_m",
    "student_teacher_fde_m",
    "student_gt_ade_m",
    "student_gt_fde_m",
    "teacher_gt_ade_m",
    "teacher_gt_fde_m",
    "goal_fde_m",
    "reverse_kl_loss",
    "x0_gap_m",
    "z1_x0_gap_m",
    "z3_x0_gap_m",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(dataset) -> str:
    """Hash effective ordered membership after train/val log filtering."""
    digest = hashlib.sha256()
    tokens = list(getattr(dataset, "tokens", []))
    paths = list(getattr(dataset, "paths", [""] * len(tokens)))
    buckets = list(getattr(dataset, "buckets", [""] * len(tokens)))
    if not tokens or len(paths) != len(tokens):
        raise ValueError("dataset_fingerprint requires ordered tokens and paths")
    for token, path, bucket in zip(tokens, paths, buckets):
        digest.update(f"{bucket}\0{token}\0{path}\n".encode("utf-8"))
    return digest.hexdigest()


class FixedOrderDistributedSampler(Sampler[int]):
    """Explicit epoch-seeded DDP sampler with auditable global ordering."""

    def __init__(
        self,
        dataset,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("invalid num_replicas/rank")

        size = len(dataset)
        if self.drop_last:
            self.num_samples = size // self.num_replicas
        else:
            self.num_samples = math.ceil(size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def global_indices(self, epoch: Optional[int] = None) -> List[int]:
        current_epoch = self.epoch if epoch is None else int(epoch)
        size = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + current_epoch)
            indices = torch.randperm(size, generator=generator).tolist()
        else:
            indices = list(range(size))
        if self.drop_last:
            return indices[: self.total_size]
        padding = self.total_size - len(indices)
        if padding > 0:
            repeats = math.ceil(padding / max(len(indices), 1))
            indices += (indices * repeats)[:padding]
        return indices

    def order_sha256(self, epoch: Optional[int] = None) -> str:
        tokens = list(getattr(self.dataset, "tokens", []))
        if len(tokens) != len(self.dataset):
            raise ValueError("sampler audit requires dataset.tokens")
        digest = hashlib.sha256()
        for index in self.global_indices(epoch):
            digest.update(str(tokens[index]).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def __iter__(self) -> Iterator[int]:
        indices = self.global_indices()
        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(rank_indices) != self.num_samples:
            raise RuntimeError("fixed sampler produced an uneven DDP shard")
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples


class SamplerAuditCallback(pl.Callback):
    """Set the sampler epoch and persist the global token-order digest."""

    def __init__(self, sampler: FixedOrderDistributedSampler, output_dir: str) -> None:
        super().__init__()
        self.sampler = sampler
        self.path = Path(output_dir) / "epoch_order_sha256.jsonl"

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        epoch = int(trainer.current_epoch)
        self.sampler.set_epoch(epoch)
        if trainer.global_rank != 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "seed": self.sampler.seed + epoch,
            "num_samples_global": self.sampler.total_size,
            "sha256": self.sampler.order_sha256(epoch),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
        print(
            f"[PhaseC-order] epoch={epoch} samples={self.sampler.total_size} "
            f"sha256={payload['sha256']}"
        )


class FixedProbeDataset(Dataset):
    """Held-out 4x32 probe addressed by absolute cache paths."""

    def __init__(
        self,
        index_path: str,
        feature_builders: Sequence[AbstractFeatureBuilder],
        target_builders: Sequence[AbstractTargetBuilder],
        *,
        viz_only: bool = False,
    ) -> None:
        super().__init__()
        self.index_path = str(index_path)
        payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        if payload.get("format") != PROBE_FORMAT:
            raise ValueError(
                f"{index_path}: expected format {PROBE_FORMAT!r}, "
                f"got {payload.get('format')!r}"
            )
        samples = list(payload.get("samples", []))
        if viz_only:
            samples = [sample for sample in samples if bool(sample.get("draw_viz"))]
        if not samples:
            raise ValueError(f"empty probe dataset: {index_path}")
        self.samples = samples
        self.tokens = [str(sample["token"]) for sample in samples]
        self.metadata = {str(sample["token"]): dict(sample) for sample in samples}
        self._feature_builders = list(feature_builders)
        self._target_builders = list(target_builders)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        token_path = Path(str(sample["path"]))
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            features.update(
                load_feature_target_from_pickle(
                    token_path / f"{builder.get_unique_name()}.gz"
                )
            )
        for builder in self._target_builders:
            targets.update(
                load_feature_target_from_pickle(
                    token_path / f"{builder.get_unique_name()}.gz"
                )
            )
        return features, targets, str(sample["token"])


def write_run_manifest(
    *,
    output_dir: str,
    repo_root: str,
    dataset_sha256: str,
    probe_index_path: str,
    init_mode: str,
    student_checkpoint: str,
    seed: int,
    world_size: int,
    batch_size: int,
    train_samples: int,
    sampler_drop_last: bool,
    sampler_samples_per_epoch: int,
    index_paths: Mapping[str, str],
    experiment_config: Mapping[str, object],
) -> Dict[str, object]:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        git_commit = "unknown"
    code_paths = (
        "navsim/planning/script/run_training_recogdrive_student_to_teacher_sft_opd.py",
        "navsim/planning/training/student_to_teacher_opd.py",
        "navsim/planning/training/direct_index_dataset.py",
        "navsim/planning/training/dataset.py",
        "navsim/planning/training/agent_lightning_module_scene_router_goal.py",
        "navsim/agents/recogdrive/recogdrive_goalbridge_goal_kd_agent.py",
        "navsim/agents/recogdrive/recogdrive_goalbridge_goal_kd_distill_trainer.py",
        "navsim/agents/recogdrive/recogdrive_goalbridge_agent.py",
        "navsim/agents/recogdrive/recogdrive_goalbridge_distill_trainer.py",
        "navsim/agents/recogdrive/recogdrive_scene_router_agent.py",
        "navsim/agents/recogdrive/recogdrive_scene_router_goal_agent.py",
        "navsim/agents/recogdrive/recogdrive_dit_scene_router_distill_trainer.py",
        "navsim/agents/recogdrive/recogdrive_diffusion_planner.py",
        "navsim/agents/recogdrive/recogdrive_predicted_goal_planner.py",
        "navsim/agents/recogdrive/recogdrive_goal_planner.py",
        "navsim/agents/recogdrive/goal_cond.py",
        "navsim/planning/script/config/common/agent/recogdrive_agent_goalbridge_goal_kd_opd.yaml",
        "navsim/planning/script/config/training/default_train_val_test_log_split.yaml",
    )
    code_sha256 = {}
    for relative in code_paths:
        code_path = Path(repo_root) / relative
        if not code_path.is_file():
            raise FileNotFoundError(f"Phase-C code file missing: {code_path}")
        code_sha256[relative] = sha256_file(code_path)
    payload: Dict[str, object] = {
        "format": "student-to-teacher-opd-run-v1",
        "git_commit": git_commit,
        "init_mode": str(init_mode),
        "student_checkpoint": str(student_checkpoint),
        "dataset_sha256": dataset_sha256,
        "probe_sha256": sha256_file(probe_index_path),
        "probe_index_path": str(probe_index_path),
        "seed": int(seed),
        "world_size": int(world_size),
        "batch_size_per_rank": int(batch_size),
        "drop_last": bool(sampler_drop_last),
        "train_samples": int(train_samples),
        "sampler_samples_per_epoch": int(sampler_samples_per_epoch),
        "index_sha256": {
            bucket: sha256_file(path) for bucket, path in index_paths.items()
        },
        "code_sha256": code_sha256,
        "experiment_config": dict(experiment_config),
    }
    path = Path(output_dir) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[PhaseC-manifest] dataset_sha256={dataset_sha256} "
        f"probe_sha256={payload['probe_sha256']}"
    )
    return payload


def _stack_seeded_noise(
    seeds: Sequence[int], shape: Tuple[int, int], device, dtype
) -> torch.Tensor:
    samples = []
    for seed in seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        samples.append(torch.randn(shape, generator=generator, dtype=torch.float32))
    return torch.stack(samples, dim=0).to(device=device, dtype=dtype)


class FixedProbeCallback(pl.Callback):
    """Run the held-out deterministic probe without perturbing training RNG."""

    def __init__(
        self,
        *,
        probe_loader: DataLoader,
        probe_dataset: FixedProbeDataset,
        output_dir: str,
        every_n_epochs: int = 1,
        interval_steps: int = 0,
    ) -> None:
        super().__init__()
        self.probe_loader = probe_loader
        self.probe_dataset = probe_dataset
        self.output_dir = Path(output_dir) / "probe_outputs"
        self.every_n_epochs = int(every_n_epochs)
        self.interval_steps = int(interval_steps)
        self._last_step = -1

    @staticmethod
    def _save_rng_state():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng_state(state) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state(state["cuda"])

    def on_fit_start(self, trainer, pl_module) -> None:
        self._run(trainer, pl_module, step=0, reason="before_opd")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if self.interval_steps > 0 and step > 0 and step % self.interval_steps == 0:
            self._run(trainer, pl_module, step=step, reason="step_interval")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.every_n_epochs > 0 and (int(trainer.current_epoch) + 1) % self.every_n_epochs == 0:
            self._run(
                trainer,
                pl_module,
                step=int(trainer.global_step),
                reason=f"epoch_{int(trainer.current_epoch)}_end",
            )

    def on_fit_end(self, trainer, pl_module) -> None:
        self._run(trainer, pl_module, step=int(trainer.global_step), reason="fit_end")

    def _run(self, trainer, pl_module, *, step: int, reason: str) -> None:
        if step == self._last_step:
            return
        self._last_step = step
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        rng_state = self._save_rng_state()
        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                local_records = self._evaluate_local(pl_module.agent)
        finally:
            pl_module.train(was_training)
            self._restore_rng_state(rng_state)

        step_dir = self.output_dir / f"step_{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        rank = int(trainer.global_rank)
        torch.save(local_records, step_dir / f"rank_{rank:02d}.pt")

        metric_rows = [
            {
                "token": record["token"],
                "bucket": record["bucket"],
                **{key: float(record[key]) for key in PROBE_METRICS},
            }
            for record in local_records
        ]
        if dist.is_available() and dist.is_initialized():
            gathered: List[Optional[List[Dict[str, object]]]] = [
                None for _ in range(dist.get_world_size())
            ]
            dist.all_gather_object(gathered, metric_rows)
            all_rows = [row for rows in gathered if rows for row in rows]
        else:
            all_rows = metric_rows

        if rank == 0:
            summary = self._summarize(all_rows)
            summary.update({"global_step": step, "reason": reason})
            (step_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tb = getattr(getattr(trainer, "logger", None), "experiment", None)
            if tb is not None and hasattr(tb, "add_scalar"):
                for key, value in summary["overall"].items():
                    tb.add_scalar(f"probe/{key}", value, global_step=step)
            print(
                f"[PhaseC-probe] step={step} reason={reason} n={len(all_rows)} "
                f"ST_ADE={summary['overall']['student_teacher_ade_m']:.4f}m "
                f"ST_FDE={summary['overall']['student_teacher_fde_m']:.4f}m "
                f"KL={summary['overall']['reverse_kl_loss']:.6f}"
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _summarize(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        def means(items):
            return {
                key: float(sum(float(row[key]) for row in items) / len(items))
                for key in PROBE_METRICS
            }

        by_bucket: Dict[str, Dict[str, float]] = {}
        for bucket in sorted({str(row["bucket"]) for row in rows}):
            selected = [row for row in rows if row["bucket"] == bucket]
            by_bucket[bucket] = means(selected)
        return {"num_samples": len(rows), "overall": means(rows), "by_bucket": by_bucket}

    def _evaluate_local(self, agent) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for features, targets, tokens in self.probe_loader:
            records.extend(self._evaluate_batch(agent, features, targets, list(tokens)))
        return records

    def _evaluate_batch(self, agent, features, targets, tokens: List[str]):
        planner = agent.action_head
        helper = agent.scene_router_trainer
        device = next(planner.parameters()).device
        student_dtype = next(planner.parameters()).dtype

        history = features["history_trajectory"].to(device)
        status = features["status_feature"].to(device)
        vl_features = features["last_hidden_state"].to(device=device, dtype=student_dtype)
        history_flat = history.reshape(history.shape[0], -1)
        gt_traj = targets["trajectory"].to(device=device, dtype=torch.float32)
        gt_goal = gt_traj[:, -1, :]
        batch_size = len(tokens)

        metadata = [self.probe_dataset.metadata[token] for token in tokens]
        buckets = [str(item["bucket"]) for item in metadata]
        noise_seeds = [int(item["noise_seed"]) for item in metadata]
        z_init = _stack_seeded_noise(
            noise_seeds,
            (planner.config.action_horizon, planner.config.action_dim),
            device,
            student_dtype,
        )

        vl_s, his_s, ego_s = helper._encode(
            planner, vl_features, history_flat, status, student_dtype
        )
        pred_goal, _ = planner.predict_goal_from_encoded(vl_s, his_s, ego_s)

        teacher_encodings = {}
        for bucket in sorted(set(buckets)):
            teacher = agent.teacher_planners[bucket]
            teacher.eval()
            teacher_encodings[bucket] = helper._encode(
                teacher, vl_features, history_flat, status, torch.float32
            )

        bucket_to_indices: Dict[str, List[int]] = {}
        for index, bucket in enumerate(buckets):
            bucket_to_indices.setdefault(bucket, []).append(index)

        canonical_z = z_init.float()
        kl_per_step: List[torch.Tensor] = []
        gap_per_step: List[torch.Tensor] = []
        intermediate: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        num_steps = int(planner.ddim_steps)

        for step_index in range(num_steps):
            t_batch = planner.make_timesteps(
                batch_size, int(planner.ddim_t[step_index].item()), device
            )
            idx_batch = planner.make_timesteps(batch_size, step_index, device)
            teacher_mean = torch.zeros_like(canonical_z)
            teacher_x0 = torch.zeros_like(canonical_z)
            with planner.goal_context(pred_goal):
                _, logvar_s, student_x0 = planner.p_mean_variance(
                    canonical_z.to(student_dtype),
                    t_batch,
                    idx_batch,
                    vl_s,
                    his_s,
                    ego_s,
                    deterministic=False,
                )

            for bucket, indices in bucket_to_indices.items():
                selected = torch.as_tensor(indices, device=device, dtype=torch.long)
                teacher = agent.teacher_planners[bucket]
                enc = teacher_encodings[bucket]
                with teacher.goal_context(gt_goal[selected]):
                    mean_t, _, x0_t = teacher.p_mean_variance(
                        canonical_z[selected].float(),
                        t_batch[selected],
                        idx_batch[selected],
                        enc[0][selected],
                        enc[1][selected],
                        enc[2][selected],
                        deterministic=True,
                    )
                teacher_mean[selected] = mean_t.float()
                teacher_x0[selected] = x0_t.float()

            sigma = (0.5 * logvar_s.float().clamp(-40.0, 20.0)).exp().detach()
            precision = helper._precision(sigma.clamp(min=helper.min_sigma).pow(2))
            mu_s = helper._shared_ddim_mean(
                planner, canonical_z, idx_batch, student_x0, sigma
            )
            mu_t = helper._shared_ddim_mean(
                planner, canonical_z, idx_batch, teacher_x0, sigma
            )
            kl_per_step.append(
                ((mu_s - mu_t).pow(2) * precision).mean(dim=(1, 2))
            )
            student_x0_m = planner.denorm_odo(student_x0.float())
            teacher_x0_m = planner.denorm_odo(teacher_x0.float())
            gap_per_step.append(
                (student_x0_m[..., :2] - teacher_x0_m[..., :2])
                .norm(dim=-1)
                .mean(dim=-1)
            )
            if step_index in (1, 3):
                intermediate[step_index] = (
                    student_x0_m.detach().cpu(),
                    teacher_x0_m.detach().cpu(),
                )
            canonical_z = teacher_mean.detach()

        teacher_traj = planner.denorm_odo(canonical_z.clamp(-1.0, 1.0)).float()
        action_input = BatchFeature(
            data={
                "his_traj": history_flat.to(student_dtype),
                "status_feature": status.to(student_dtype),
            }
        )
        with planner.goal_context(pred_goal):
            student_out = planner.get_action(
                vl_features,
                action_input,
                init_actions=z_init,
                deterministic=True,
            )
        student_traj = student_out["pred_traj"].float()

        st_error = (student_traj[..., :2] - teacher_traj[..., :2]).norm(dim=-1)
        sg_error = (student_traj[..., :2] - gt_traj[..., :2]).norm(dim=-1)
        tg_error = (teacher_traj[..., :2] - gt_traj[..., :2]).norm(dim=-1)
        goal_error = (pred_goal.float()[..., :2] - gt_goal[..., :2]).norm(dim=-1)
        mean_kl = torch.stack(kl_per_step, dim=0).mean(dim=0)
        mean_gap = torch.stack(gap_per_step, dim=0).mean(dim=0)

        if 1 not in intermediate or 3 not in intermediate:
            raise RuntimeError(
                f"probe expects at least four DDIM steps, got {num_steps}"
            )
        z1_student, z1_teacher = intermediate[1]
        z3_student, z3_teacher = intermediate[3]
        z1_gap = (
            (z1_student[..., :2] - z1_teacher[..., :2]).norm(dim=-1).mean(dim=-1)
        )
        z3_gap = (
            (z3_student[..., :2] - z3_teacher[..., :2]).norm(dim=-1).mean(dim=-1)
        )

        records = []
        for index, token in enumerate(tokens):
            meta = metadata[index]
            records.append(
                {
                    "token": token,
                    "bucket": buckets[index],
                    "noise_seed": noise_seeds[index],
                    "draw_viz": bool(meta.get("draw_viz")),
                    "gt_traj": gt_traj[index].detach().cpu(),
                    "teacher_traj": teacher_traj[index].detach().cpu(),
                    "student_traj": student_traj[index].detach().cpu(),
                    "gt_goal": gt_goal[index].detach().cpu(),
                    "pred_goal": pred_goal[index].float().detach().cpu(),
                    "z1_teacher_x0": z1_teacher[index],
                    "z1_student_x0": z1_student[index],
                    "z3_teacher_x0": z3_teacher[index],
                    "z3_student_x0": z3_student[index],
                    "student_teacher_ade_m": float(st_error[index].mean()),
                    "student_teacher_fde_m": float(st_error[index, -1]),
                    "student_gt_ade_m": float(sg_error[index].mean()),
                    "student_gt_fde_m": float(sg_error[index, -1]),
                    "teacher_gt_ade_m": float(tg_error[index].mean()),
                    "teacher_gt_fde_m": float(tg_error[index, -1]),
                    "goal_fde_m": float(goal_error[index]),
                    "reverse_kl_loss": float(mean_kl[index]),
                    "x0_gap_m": float(mean_gap[index]),
                    "z1_x0_gap_m": float(z1_gap[index]),
                    "z3_x0_gap_m": float(z3_gap[index]),
                }
            )
        return records
