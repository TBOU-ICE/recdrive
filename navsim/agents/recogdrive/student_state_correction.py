"""Fixed student-state correction utilities. No VLM loading, GT only for teacher."""
from __future__ import annotations
import contextlib
import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


def load_pt(path):
    # Only load trusted local project checkpoints/caches.
    return torch.load(path, map_location="cpu", weights_only=False)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_for(token, seed, repeat=0):
    return int.from_bytes(hashlib.sha256(f"{seed}:{token}:{repeat}".encode()).digest()[:4], "big")


def read_indexes(paths):
    """Read the project's direct-index format; reject conflicting duplicate tokens."""
    entries = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        if sorted(payload.get("builders", [])) != ["internvl_feature", "trajectory_target"]:
            raise ValueError(f"{path}: expected internvl_feature and trajectory_target builders")
        rows = payload.get("samples")
        if rows is None:
            root = Path(payload["cache_path"])
            rows = [{"token": t, "path": str(root / p)} for t, p in payload["tokens"].items()]
        for row in rows:
            token, folder = str(row["token"]), str(Path(row["path"]).expanduser())
            if token in entries and entries[token] != folder:
                raise ValueError(f"Conflicting cache paths for {token}")
            entries[token] = folder
    if not entries:
        raise ValueError("Empty direct indexes")
    return [{"token": t, "path": p} for t, p in sorted(entries.items())]


def assert_disjoint(train, val):
    # Require separate logs as well as tokens: adjacent frames must not leak.
    train_tokens = {r["token"] for r in train}
    val_tokens = {r["token"] for r in val}
    if train_tokens & val_tokens:
        raise ValueError("Train/validation token overlap")
    train_logs = {Path(r["path"]).parent.name for r in train}
    val_logs = {Path(r["path"]).parent.name for r in val}
    if train_logs & val_logs:
        raise ValueError("Train/validation log overlap; split indexes by log, not by frame")


def raw_sample(row):
    folder = Path(row["path"])
    with gzip.open(folder / "internvl_feature.gz", "rb") as f:
        features = pickle.load(f)
    with gzip.open(folder / "trajectory_target.gz", "rb") as f:
        targets = pickle.load(f)
    return {"token": row["token"], "path": row["path"],
            "vl": features["last_hidden_state"].float().cpu(),
            "history": features["history_trajectory"].float().reshape(-1).cpu(),
            "status": features["status_feature"].float().cpu(),
            "gt": targets["trajectory"].float().cpu()}


def inputs(sample, device):
    vl = sample["vl"].to(device).unsqueeze(0)
    history = sample["history"].to(device).unsqueeze(0)
    status = sample["status"].to(device).unsqueeze(0)
    gt = sample["gt"].to(device).unsqueeze(0)
    return vl, BatchFeature(data={"his_traj": history, "status_feature": status,
                                  "action": gt, "goal": gt[:, -1]})


def encode(planner, vl, action):
    return (planner.feature_encoder(vl),
            planner.his_traj_encoder(action.his_traj.unsqueeze(1)).repeat(1, planner.config.action_horizon, 1),
            planner.ego_status_encoder(action.status_feature))


def planner_state(checkpoint):
    state = checkpoint.get("state_dict", checkpoint)
    # Select the student's/action-head branch, never frozen teacher or EMA branches.
    for prefix in ("module.agent.action_head.", "agent.action_head.", "module.action_head.", "action_head."):
        selected = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if selected:
            return selected
    return {k.removeprefix("module."): v for k, v in state.items()}


def build_planner(spec, device):
    from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner
    from navsim.agents.recogdrive.recogdrive_goal_planner import GoalCondDiffusionPlanner
    from navsim.agents.recogdrive.recogdrive_predicted_goal_planner import PredictedGoalDiffusionPlanner
    size = spec.get("dit_type", "small")
    cfg = make_recogdrive_config(size, action_dim=3, action_horizon=8,
                                input_embedding_dim=384 if size == "small" else 1536,
                                sampling_method="ddim", num_inference_steps=spec.get("steps", 5),
                                model_dtype="float32")
    cfg.vlm_size = spec.get("vlm_size", "small")
    kind = spec.get("kind", "goal_free")
    if kind == "goal_free":
        planner = ReCogDriveDiffusionPlanner(cfg)
    else:
        cls = {"privileged": GoalCondDiffusionPlanner, "pred_goal": PredictedGoalDiffusionPlanner}[kind]
        kwargs = dict(spec.get("goal_kwargs", {}))
        kwargs["goal_mode"] = spec.get("goal_mode", "adaln")
        planner = cls(cfg, **kwargs)
    state = planner_state(load_pt(spec["checkpoint"]))
    # Require all learned parameters. Do not silently drop goal heads.
    expected = dict(planner.named_parameters())
    missing = [k for k in expected if k not in state]
    mismatch = [k for k in expected if k in state and state[k].shape != expected[k].shape]
    unexpected = set(state) - set(planner.state_dict())
    if missing or mismatch or unexpected:
        raise ValueError(f"Checkpoint/model mismatch: missing={missing}, shapes={mismatch}, extra={sorted(unexpected)}")
    # Schedules are checkpoint buffers: confirm they agree with the configured steps.
    for k in ("ddim_t", "ddim_alphas", "ddim_alphas_prev"):
        if k in state and not torch.equal(state[k].cpu(), planner.state_dict()[k].cpu()):
            raise ValueError(f"Checkpoint schedule mismatch: {k}")
    planner.load_state_dict(state, strict=False)
    planner.model.set_adaln_bound(float(spec.get("adaln_bound", 0.0)))
    for k, v in spec.get("sampling_attributes", {}).items():
        if k not in {"denoised_clip_value", "eps_clip_value", "final_action_clip_value"}:
            raise ValueError(f"Unsupported sampling attribute {k}")
        setattr(planner, k, v)
    return planner.float().to(device)


def goal_context(planner, enc, gt, privileged):
    if privileged:
        return planner.goal_context(gt[:, -1])
    if hasattr(planner, "predict_goal_from_encoded"):
        return planner.goal_context(planner.predict_goal_from_encoded(*enc)[0])
    return contextlib.nullcontext()


@torch.no_grad()
def rollout(planner, vl, action, sampler, privileged=False):
    """Store states BEFORE every DiT call. Last state is z4, not final output."""
    enc = encode(planner, vl, action)
    z = torch.randn_like(action.action)
    states = []
    with goal_context(planner, enc, action.action, privileged):
        for step, timestep in enumerate(planner.ddim_t):
            states.append(z.clone())
            t = planner.make_timesteps(len(z), int(timestep), z.device)
            idx = planner.make_timesteps(len(z), step, z.device)
            mu, logvar, x0 = planner.p_mean_variance(z, t, idx, *enc, deterministic=sampler["deterministic"])
            sigma = (0.5 * logvar).exp()
            sigma = torch.zeros_like(sigma) if sampler["deterministic"] else sigma.clamp_min(sampler["min_sigma"])
            noise = torch.randn_like(z).clamp(-sampler["noise_clip"], sampler["noise_clip"])
            z = mu + sigma * noise
    clip = getattr(planner, "final_action_clip_value", 1.0)
    final = z if clip is None else z.clamp(-clip, clip)
    return states, planner.denorm_odo(final), x0


def teacher_prediction(planner, z, step, enc, goal):
    """Expose raw epsilon/x0 for diagnostics, use production clipping for the loss."""
    t = planner.make_timesteps(len(z), int(planner.ddim_t[step]), z.device)
    idx = planner.make_timesteps(len(z), step, z.device)
    embedding = planner._goal_embedding_for(goal, z.dtype)
    eps = planner._dit_step(z, t, *enc, embedding)
    alpha = planner.extract(planner.ddim_alphas, idx, z.shape)
    noise_scale = planner.extract(planner.ddim_sqrt_one_minus_alphas, idx, z.shape)
    raw = (z - noise_scale * eps) / alpha.sqrt()
    clip = getattr(planner, "denoised_clip_value", 1.0)
    x0 = raw.clamp(-clip, clip)
    return x0, raw, eps


def errors(pred, gt):
    d = (pred[..., :2] - gt[..., :2]).norm(dim=-1)
    return {"ade_m": float(d.mean()), "fde_m": float(d[..., -1].mean())}


def grad_norm(parameters):
    parts = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(parts).sum().sqrt()) if parts else 0.0


def branch_gradient_diagnostics(correction_loss, retention_loss, planner, correction_weight, retention_weight):
    """Compare the two objectives on shared parameters without accumulating .grad.

    These diagnostics are intentionally computed only at the configured
    diagnostic interval.  The returned gradients are local-rank batch
    gradients; the subsequent total-loss backward remains the DDP-synchronised
    update used by the optimizer.
    """
    named = [(name, p) for name, p in planner.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    corr_grads = torch.autograd.grad(
        correction_loss, params, retain_graph=True, allow_unused=True
    )
    retain_grads = torch.autograd.grad(
        retention_loss, params, retain_graph=True, allow_unused=True
    )

    def stats(prefix, selected):
        corr_sq = correction_loss.new_zeros((), dtype=torch.float32)
        retain_sq = correction_loss.new_zeros((), dtype=torch.float32)
        dot = correction_loss.new_zeros((), dtype=torch.float32)
        jointly_active = correction_loss.new_zeros((), dtype=torch.float32)
        conflicting = correction_loss.new_zeros((), dtype=torch.float32)
        for idx in selected:
            corr = corr_grads[idx]
            retain = retain_grads[idx]
            if corr is None and retain is None:
                continue
            if corr is None:
                corr = torch.zeros_like(retain)
            if retain is None:
                retain = torch.zeros_like(corr)
            corr = corr.detach().float()
            retain = retain.detach().float()
            corr_sq = corr_sq + corr.square().sum()
            retain_sq = retain_sq + retain.square().sum()
            product = corr * retain
            dot = dot + product.sum()
            jointly_active = jointly_active + (product != 0).sum()
            conflicting = conflicting + (product < 0).sum()
        corr_norm = corr_sq.sqrt()
        retain_norm = retain_sq.sqrt()
        cosine = dot / (corr_norm * retain_norm).clamp_min(1e-12)
        weighted_corr = corr_norm * abs(float(correction_weight))
        weighted_retain = retain_norm * abs(float(retention_weight))
        ratio = weighted_corr / weighted_retain.clamp_min(1e-12)
        return {
            f"{prefix}correction_grad_norm": float(corr_norm),
            f"{prefix}retention_grad_norm": float(retain_norm),
            f"{prefix}weighted_correction_grad_norm": float(weighted_corr),
            f"{prefix}weighted_retention_grad_norm": float(weighted_retain),
            f"{prefix}weighted_grad_norm_ratio": float(ratio),
            f"{prefix}gradient_dot": float(dot),
            f"{prefix}gradient_cosine": float(cosine),
            f"{prefix}gradient_conflict_fraction": float(
                conflicting / jointly_active.clamp_min(1)
            ),
        }

    all_indices = list(range(len(named)))
    decoder_indices = [
        idx for idx, (name, _) in enumerate(named) if name.startswith("action_decoder.")
    ]
    output = stats("branches/", all_indices)
    output.update(stats("decoder_branches/", decoder_indices))
    return output


def summary(rows):
    out = {"n": len(rows)}
    for key in sorted({k for r in rows for k, v in r.items() if k not in {"seed", "step"} and isinstance(v, (int, float)) and not isinstance(v, bool)}):
        observed = [r[key] for r in rows if key in r]
        if any(not math.isfinite(v) for v in observed):
            raise FloatingPointError(f"Non-finite metric: {key}")
        values = torch.tensor(observed, dtype=torch.float64)
        if len(values):
            out[key] = {"n": len(values), "mean": values.mean().item(),
                        "median": values.quantile(.5).item(), "p90": values.quantile(.9).item()}
    return out


class CorrectionObjective(nn.Module):
    def __init__(self, teacher, cfg):
        super().__init__()
        self.planner, self.cfg = teacher, cfg

    def forward(self, samples, diagnostic=False):
        correction_losses, retention_losses, metrics = [], [], []
        device = next(self.planner.parameters()).device
        for sample in samples:
            p = self.planner
            vl, action = inputs(sample, device)
            enc = encode(p, vl, action)
            xgt = p.norm_odo(action.action)
            correction = []
            for entry in sample["states"]:
                z, step = entry["z"].to(device).unsqueeze(0), entry["step"]
                # Clean goal for correction; robust noise/dropout only in retention.
                old = p.goal_dropout_p, p.goal_noise_p
                p.goal_dropout_p = p.goal_noise_p = 0.0
                try:
                    x0, raw, eps = teacher_prediction(p, z, step, enc, action.goal)
                finally:
                    p.goal_dropout_p, p.goal_noise_p = old
                loss = (x0 - xgt).square().mean()
                correction.append(loss)
                row = {f"z{step}/corr_loss": float(loss.detach()),
                       f"z{step}/eps_abs_mean": float(eps.detach().abs().mean()),
                       f"z{step}/clip_fraction": float((raw.detach().abs() >= getattr(p, "denoised_clip_value", 1.0)).float().mean())}
                with torch.no_grad():
                    pred, before = p.denorm_odo(x0), p.denorm_odo(z)
                    row.update({f"z{step}/{k}": v for k, v in errors(pred, action.action).items()})
                    row[f"z{step}/correction_m"] = float((pred[..., :2] - before[..., :2]).norm(dim=-1).mean())
                    row[f"z{step}/ade_gain_m"] = errors(before, action.action)["ade_m"] - row[f"z{step}/ade_m"]
                if diagnostic:
                    g = torch.autograd.grad(loss, eps, retain_graph=True)[0]
                    row[f"z{step}/corr_grad_eps_norm"] = float(g.norm())
                metrics.append(row)
            corr_loss = torch.stack(correction).mean()
            # Every update has one correction and one standard q(GT) loss.
            # This is a 1:1 pair of objectives, not a 50% chance of omitting one.
            retain_loss = p(vl, action)["loss"]
            correction_losses.append(corr_loss)
            retention_losses.append(retain_loss)
            metrics.append({"correction_loss": float(corr_loss.detach()), "retention_eps_loss": float(retain_loss.detach())})
        corr_batch = torch.stack(correction_losses).mean()
        retain_batch = torch.stack(retention_losses).mean()
        if diagnostic:
            metrics.append(branch_gradient_diagnostics(
                corr_batch,
                retain_batch,
                self.planner,
                self.cfg["correction_weight"],
                self.cfg["retention_weight"],
            ))
        total = (
            self.cfg["correction_weight"] * corr_batch
            + self.cfg["retention_weight"] * retain_batch
        )
        return total, metrics


class OPDObjective(nn.Module):
    def __init__(self, student, teacher, cfg, sampler):
        super().__init__()
        self.planner = student
        self.teacher = teacher.eval().requires_grad_(False)
        if hasattr(student, "predict_goal_from_encoded"):
            # The state-correction experiment is intended to preserve the
            # deployable predicted-goal student.  The student conditions its
            # rollout on its own predicted goal; only the frozen privileged
            # teacher receives the GT goal.
            from navsim.agents.recogdrive.recogdrive_goalbridge_goal_kd_distill_trainer import (
                ReCogDriveGoalBridgeGoalKDDistillTrainer,
            )
            self.trainer = ReCogDriveGoalBridgeGoalKDDistillTrainer(
                bucket_names=["single"], fallback_bucket="single",
                collect_viz=False, min_sigma=sampler["min_sigma"],
                goal_loss_weight=cfg.get("goal_loss_weight", 1.0),
                kd_weight=cfg["kd_weight"],
                kl_precision_clip=cfg["precision_clip"],
            )
        else:
            from navsim.agents.recogdrive.recogdrive_goalfree_kd_only_distill_trainer import (
                ReCogDriveGoalFreeKDOnlyDistillTrainer,
            )
            self.trainer = ReCogDriveGoalFreeKDOnlyDistillTrainer(
                bucket_names=["single"], fallback_bucket="single", collect_viz=False,
                kd_weight=cfg["kd_weight"], min_sigma=cfg["loss_min_sigma"],
                kl_precision_clip=cfg["precision_clip"],
                rollout_min_sigma=sampler["min_sigma"], rollout_noise_clip=sampler["noise_clip"],
                rollout_deterministic=sampler["deterministic"], fail_on_nonfinite=True)

    def forward(self, samples, diagnostic=False):
        self.teacher.eval()
        self.trainer.diagnostic_gradients = diagnostic
        losses, rows = [], []
        for sample in samples:
            vl, action = inputs(sample, next(self.planner.parameters()).device)
            result = self.trainer.compute_loss(self.planner, {"single": self.teacher}, None, vl, action, ["single"])
            losses.append(result["loss"])
            rows.append({k: float(v.detach()) for k, v in result.items() if k != "loss" and isinstance(v, torch.Tensor) and v.numel() == 1})
        return torch.stack(losses).mean(), rows


class PDMScoring:
    def __init__(self, cache_path):
        from navsim.common.dataloader import MetricCacheLoader
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        self.cache = MetricCacheLoader(Path(cache_path))
        sampling = TrajectorySampling(num_poses=40, interval_length=.1)
        self.simulator = PDMSimulator(sampling)
        self.scorer = PDMScorer(sampling, PDMScorerConfig())

    def __call__(self, token, trajectory):
        import lzma
        from dataclasses import asdict
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import pdm_score
        with lzma.open(self.cache.metric_cache_paths[token], "rb") as f:
            cache = pickle.load(f)
        result = pdm_score(metric_cache=cache,
                           model_trajectory=Trajectory(trajectory.detach().cpu().numpy()),
                           future_sampling=self.simulator.proposal_sampling,
                           simulator=self.simulator, scorer=self.scorer)
        return asdict(result)
