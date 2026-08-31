"""Goal-adapter planners for Privileged-OPD v2.

Additive implementation: existing ReCogDrive planners/scripts are untouched.

The planner is warm-started from a goal-free IL/RL checkpoint. Privileged
information is injected only through a small goal branch so that Goal OFF stays
close to the original policy. Supported injection modes:

- ``adaln``: goal summary is added to the normal AdaLN conditioning (ablation).
- ``cross``: goal tokens are appended to cached VLM K/V context (ablation).
- ``gated_cross``: action tokens query goal tokens through a separate residual
  cross-attention adapter with a scalar gate initialized to exactly zero. This
  is the recommended default because the initial policy is *exactly* the base
  policy and the privileged branch is learned as a residual.

Goal representation can be either a single final point or a 3-point route
(near/mid/far).  Cached VLM hidden states remain reusable: all goal operations
happen inside the DiT planner after the cache boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Sequence

import torch
from torch import nn

from navsim.agents.recogdrive.goal_cond import GoalEncoder
from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)

GOAL_INJECTION_MODES = ("adaln", "cross", "gated_cross")
GOAL_POINT_MODES = ("final", "multi3")


def extract_goal_points(
    trajectory: torch.Tensor,
    mode: str = "final",
    indices: Sequence[int] = (1, 4, 7),
) -> torch.Tensor:
    """Extract privileged point(s) from an 8-point GT trajectory.

    Returns ``(B, M, 3)`` where M=1 for ``final`` and M=3 for ``multi3``.
    The default multi-point positions correspond approximately to 1s, 2.5s,
    and 4s for the project's 0.5s trajectory interval.
    """
    if trajectory.ndim != 3 or trajectory.shape[-1] < 3:
        raise ValueError(f"trajectory must have shape (B,H,>=3), got {tuple(trajectory.shape)}")
    if mode not in GOAL_POINT_MODES:
        raise ValueError(f"goal_point_mode must be one of {GOAL_POINT_MODES}, got {mode!r}")
    if mode == "final":
        return trajectory[:, -1:, :3].contiguous()
    if len(indices) != 3:
        raise ValueError(f"multi3 requires exactly three indices, got {indices}")
    h = trajectory.shape[1]
    idx = [i if i >= 0 else h + i for i in indices]
    if any(i < 0 or i >= h for i in idx):
        raise IndexError(f"goal indices {indices} out of range for horizon={h}")
    return trajectory[:, idx, :3].contiguous()


class PrivilegedGoalAdapterPlanner(ReCogDriveDiffusionPlanner):
    """Goal-free planner + a lightweight privileged residual branch."""

    def __init__(
        self,
        config: ReCogDriveDiffusionPlannerConfig,
        goal_injection: str = "gated_cross",
        goal_point_mode: str = "final",
        goal_indices: Sequence[int] = (1, 4, 7),
        goal_sincos_dim: int = 128,
        goal_hidden_dim: int = 512,
        goal_use_heading: bool = False,
        goal_adapter_heads: int = 8,
    ):
        if goal_injection not in GOAL_INJECTION_MODES:
            raise ValueError(f"goal_injection must be one of {GOAL_INJECTION_MODES}, got {goal_injection!r}")
        if goal_point_mode not in GOAL_POINT_MODES:
            raise ValueError(f"goal_point_mode must be one of {GOAL_POINT_MODES}, got {goal_point_mode!r}")
        if config.sampling_method != "ddim":
            raise NotImplementedError("PrivilegedOPD-v2 currently requires DDIM.")
        super().__init__(config)

        self.goal_injection = str(goal_injection)
        self.goal_point_mode = str(goal_point_mode)
        self.goal_indices = tuple(int(x) for x in goal_indices)
        self.goal_use_heading = bool(goal_use_heading)
        self._goal_ctx: Optional[torch.Tensor] = None

        dim = int(config.input_embedding_dim)
        # For gated_cross the residual gate, rather than the encoder, supplies the
        # exact no-op initialization. For cross we also keep the encoder normally
        # initialized so the branch is trainable immediately.
        self.goal_encoder = GoalEncoder(
            out_dim=dim,
            hidden_dim=int(goal_hidden_dim),
            sincos_dim=int(goal_sincos_dim),
            use_heading=self.goal_use_heading,
            zero_init_last=(goal_injection == "adaln"),
        )
        # Zero type embeddings preserve the base policy for AdaLN at init; they
        # receive gradient immediately and can separate near/mid/far during training.
        self.goal_type_embed = nn.Parameter(torch.zeros(3, dim))

        if goal_injection == "cross":
            # Deliberately not zero-init: this is the direct K/V-token ablation.
            # It does not preserve the base policy exactly at initialization.
            self.goal_cross_proj = nn.Linear(dim, dim)
        elif goal_injection == "gated_cross":
            if dim % goal_adapter_heads != 0:
                raise ValueError(f"embed dim {dim} must be divisible by goal_adapter_heads={goal_adapter_heads}")
            self.goal_query_norm = nn.LayerNorm(dim)
            self.goal_kv_norm = nn.LayerNorm(dim)
            self.goal_cross_adapter = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=int(goal_adapter_heads),
                batch_first=True,
            )
            # Direct scalar gate (not sigmoid): alpha=0 makes the adapter an exact
            # no-op while still giving alpha a non-zero first-step gradient.
            self.goal_adapter_gate = nn.Parameter(torch.zeros(()))

    @contextmanager
    def goal_context(self, goal_points: Optional[torch.Tensor]):
        prev = self._goal_ctx
        self._goal_ctx = goal_points
        try:
            yield self
        finally:
            self._goal_ctx = prev

    def _resolve_goal(self, goal: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
        g = goal if goal is not None else self._goal_ctx
        if g is None:
            return None
        if g.ndim == 2:
            g = g.unsqueeze(1)
        if g.ndim != 3:
            raise ValueError(f"goal points must be (B,M,3) or (B,3), got {tuple(g.shape)}")
        if g.shape[0] == batch_size:
            return g
        if batch_size % g.shape[0] == 0:
            return g.repeat_interleave(batch_size // g.shape[0], dim=0)
        raise ValueError(f"Cannot broadcast goal batch={g.shape[0]} to batch={batch_size}")

    def encode_goal_tokens(self, goal_points: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if goal_points.ndim == 2:
            goal_points = goal_points.unsqueeze(1)
        b, m, _ = goal_points.shape
        if m > 3:
            raise ValueError(f"At most 3 privileged points are supported, got M={m}")
        # norm_odo accepts (..., H, 3), so the point set can be normalized as a
        # tiny trajectory without changing the project's coordinate convention.
        goal_norm = self.norm_odo(goal_points[..., :3]).to(dtype)
        flat = goal_norm.reshape(b * m, 3)
        emb = self.goal_encoder(flat).reshape(b, m, -1).to(dtype)
        emb = emb + self.goal_type_embed[:m].unsqueeze(0).to(dtype)
        return emb

    def _dit_step_with_goal(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        vl_embeds: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        goal_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        action_features = self.action_encoder(actions, timesteps)
        if hasattr(self, "position_embedding"):
            pos_ids = torch.arange(action_features.shape[1], device=actions.device)
            action_features = action_features + self.position_embedding(pos_ids)

        vl_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
        fused = self.fusion_projector(torch.cat((his_traj_features, vl_mean, action_features), dim=2))
        encoder_states = vl_embeds
        conditioning = ego_status_features

        if goal_tokens is not None:
            if self.goal_injection == "adaln":
                conditioning = conditioning + goal_tokens.mean(dim=1)
            elif self.goal_injection == "cross":
                g = self.goal_cross_proj(goal_tokens).to(vl_embeds.dtype)
                encoder_states = torch.cat([vl_embeds, g], dim=1)
            elif self.goal_injection == "gated_cross":
                q = self.goal_query_norm(fused)
                kv = self.goal_kv_norm(goal_tokens.to(fused.dtype))
                residual, _ = self.goal_cross_adapter(q, kv, kv, need_weights=False)
                fused = fused + self.goal_adapter_gate.to(fused.dtype) * residual

        out = self.model(
            hidden_states=fused,
            encoder_hidden_states=encoder_states,
            conditioning_features=conditioning,
            timesteps=timesteps,
        )
        return self.action_decoder(out)

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        index: torch.Tensor,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        deterministic: bool = True,
        goal: Optional[torch.Tensor] = None,
    ):
        goal_points = self._resolve_goal(goal, x.shape[0])
        goal_tokens = None if goal_points is None else self.encode_goal_tokens(goal_points, x.dtype)

        model_dtype = next(self.model.parameters()).dtype
        x = x.to(model_dtype)
        model_output = self._dit_step_with_goal(
            x, t, vl_features, his_traj_features, ego_status_features, goal_tokens
        )
        alpha_t = self.extract(self.ddim_alphas, index, x.shape)
        sqrt_one_minus_alpha_t = self.extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
        x_recon = (x - sqrt_one_minus_alpha_t * model_output) / alpha_t.sqrt()
        denoised_clip_value = getattr(self, "denoised_clip_value", 1.0)
        x_recon = x_recon.clamp(-float(denoised_clip_value), float(denoised_clip_value))

        alpha_prev = self.extract(self.ddim_alphas_prev, index, x.shape)
        pred_noise = (x - alpha_t.sqrt() * x_recon) / sqrt_one_minus_alpha_t.clamp(min=1e-8)
        eps_clip_value = getattr(self, "eps_clip_value", None)
        if eps_clip_value is not None:
            pred_noise = pred_noise.clamp(-float(eps_clip_value), float(eps_clip_value))
        if deterministic:
            etas = torch.zeros((x.shape[0], 1, 1), device=x.device, dtype=x.dtype)
        else:
            etas = self.eta(x).unsqueeze(1)
        sigma = (
            etas
            * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).clamp(min=0).sqrt()
        ).clamp(min=1e-10)
        pred_dir = (1.0 - alpha_prev - sigma ** 2).clamp(min=0).sqrt() * pred_noise
        model_mean = alpha_prev.sqrt() * x_recon + pred_dir
        model_log_variance = torch.log(sigma ** 2 + 1e-20)
        return model_mean, model_log_variance, x_recon

    def forward(self, vl_features: torch.Tensor, action_input):
        """Standard diffusion IL loss with privileged goal points."""
        goal = None
        data = getattr(action_input, "data", None)
        if isinstance(data, dict):
            goal = data.get("goal_points", data.get("goal"))
        if goal is None:
            goal = getattr(action_input, "goal_points", None)
        if goal is None:
            raise RuntimeError("PrivilegedGoalAdapterPlanner.forward requires goal_points")

        vl_embeds = self.feature_encoder(vl_features)
        his = self.his_traj_encoder(action_input.his_traj.unsqueeze(1)).repeat(
            1, self.config.action_horizon, 1
        )
        ego = self.ego_status_encoder(action_input.status_feature)
        gt = self.norm_odo(action_input.action)
        noise = torch.randn_like(gt)
        t = self.sample_time(gt.shape[0], device=gt.device, dtype=gt.dtype)
        noisy = (
            self.extract(self.ddpm_sqrt_alphas_cumprod, t, gt.shape) * gt
            + self.extract(self.ddpm_sqrt_one_minus_alphas_cumprod, t, gt.shape) * noise
        )
        goal_tokens = self.encode_goal_tokens(self._resolve_goal(goal, gt.shape[0]), gt.dtype)
        pred_noise = self._dit_step_with_goal(noisy, t, vl_embeds, his, ego, goal_tokens)
        loss = torch.nn.functional.mse_loss(pred_noise, noise, reduction="mean")
        from transformers.feature_extraction_utils import BatchFeature
        return BatchFeature(data={"loss": loss})

    def goal_parameter_names(self):
        prefixes = (
            "goal_encoder.",
            "goal_type_embed",
            "goal_cross_proj.",
            "goal_query_norm.",
            "goal_kv_norm.",
            "goal_cross_adapter.",
            "goal_adapter_gate",
        )
        return [n for n, _ in self.named_parameters() if n.startswith(prefixes)]
