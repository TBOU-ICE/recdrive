"""Goal-conditioned diffusion planner for privileged-teacher training.

Subclasses :class:`ReCogDriveDiffusionPlanner` without modifying it, so every
existing script keeps its current behaviour.  Four goal modes are available:

``inpaint``
    Training-free diagnostic.  The predicted clean trajectory ``x_recon`` has its
    endpoint overwritten with the ground-truth goal at every denoising step
    (reconstruction guidance).  Use this to measure how much headroom a perfect
    goal buys before investing in any training.

``adaln``
    The goal embedding is added to the AdaLN conditioning vector, next to the
    ego status.  Note that ``driving_command`` (turn left / go straight /
    turn right) already enters through this exact door as part of
    ``status_feature``, and the goal point is its continuous refinement.
    This is the only mode that needs no plumbing changes in the OPD distillation
    trainer, because the goal is folded into ``ego_status_features`` upstream of
    ``p_mean_variance``'s output.

``channel``
    The goal embedding is added into the residual stream alongside
    ``fusion_projector``'s output.  Equivalent to widening ``fusion_projector``
    from ``Linear(3D -> D)`` to ``Linear(4D -> D)``, but implemented as a
    separate zero-initialised projection so the pretrained weights are preserved
    bit-for-bit and no checkpoint migration is needed.

``cross``
    The goal embedding is appended as one extra key/value token to the
    cross-attention context.  Note that ``interleave_attention=True`` means only
    the odd-indexed DiT blocks run cross-attention, so this mode reaches 8 of the
    16 layers.  The mean used by the channel path is computed from the original
    VLM features, so the two paths stay independent.

The goal itself is the ground-truth trajectory endpoint and is passed either
explicitly or through :meth:`GoalCondDiffusionPlanner.goal_context`, which lets
call sites that this class does not override (``sample_chain``, ``get_logprobs``,
and the distillation trainer) pick it up without any signature change.
"""

from __future__ import annotations

import copy
import inspect
from contextlib import contextmanager
from typing import Optional

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from .goal_cond import GOAL_MODES, TRAINABLE_GOAL_MODES, GoalEncoder, zero_init_linear
from .recogdrive_diffusion_planner import (
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


def extract_goal(action_input, batch_size: Optional[int] = None) -> Optional[torch.Tensor]:
    """Pull a ``goal`` entry out of a ``BatchFeature`` / dict, tolerating absence."""
    if action_input is None:
        return None
    data = getattr(action_input, "data", None)
    if isinstance(data, dict):
        return data.get("goal")
    if isinstance(action_input, dict):
        return action_input.get("goal")
    return getattr(action_input, "goal", None)


class GoalCondDiffusionPlanner(ReCogDriveDiffusionPlanner):
    """Diffusion planner that can be conditioned on the ground-truth goal point."""

    def __init__(
        self,
        config: ReCogDriveDiffusionPlannerConfig,
        goal_mode: str = "none",
        goal_sincos_dim: int = 128,
        goal_hidden_dim: int = 1024,
        goal_use_heading: bool = False,
        goal_dropout_p: float = 0.0,
        goal_noise_p: float = 0.0,
        goal_noise_std_xy: float = 0.0,
        goal_noise_std_heading: float = 0.0,
        goal_inpaint_weight: float = 1.0,
        goal_inpaint_heading: bool = False,
    ):
        if goal_mode not in GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {GOAL_MODES}, got {goal_mode!r}")

        # Plain attributes assigned before nn.Module.__init__ fall through to
        # object.__setattr__, which is what lets us defer the GRPO setup below.
        self._goal_defer_grpo = True

        super().__init__(config)

        self.goal_mode = goal_mode
        self.goal_dropout_p = float(goal_dropout_p)
        self.goal_noise_p = float(goal_noise_p)
        self.goal_noise_std_xy = float(goal_noise_std_xy)
        self.goal_noise_std_heading = float(goal_noise_std_heading)
        if self.goal_dropout_p < 0.0 or self.goal_noise_p < 0.0 or self.goal_dropout_p + self.goal_noise_p > 1.0:
            raise ValueError(
                f"goal_dropout_p + goal_noise_p must be in [0, 1], got "
                f"{self.goal_dropout_p} + {self.goal_noise_p}"
            )
        self.goal_inpaint_weight = float(goal_inpaint_weight)
        self.goal_inpaint_heading = bool(goal_inpaint_heading)
        self._goal_ctx: Optional[torch.Tensor] = None

        if goal_mode in TRAINABLE_GOAL_MODES and config.sampling_method == "flow":
            raise NotImplementedError(
                "Goal conditioning is wired for the ddpm/ddim samplers. The 'flow' "
                "branches of get_action/sample_chain bypass p_mean_variance and are "
                "not covered; switch sampling_method to 'ddim'."
            )

        embed_dim = config.input_embedding_dim
        if goal_mode in TRAINABLE_GOAL_MODES:
            # Exactly ONE zero-initialised layer per goal branch, at the OUTERMOST
            # position (ControlNet zero-conv rule).  For adaln the encoder output is
            # consumed directly, so its last layer is the zero layer.  For channel and
            # cross the zero layer is the projection below; the encoder must then be
            # normally initialised, because two stacked zero layers block each other's
            # gradients (dL/dP = g.e^T = 0 and dL/de = P^T.g = 0) and the goal branch
            # would stay dead forever -- confirmed on the 2026.08.02 channel/cross
            # teacher checkpoints, whose goal weights were still exactly zero after
            # ~200 epochs.
            self.goal_encoder = GoalEncoder(
                out_dim=embed_dim,
                hidden_dim=goal_hidden_dim,
                sincos_dim=goal_sincos_dim,
                use_heading=goal_use_heading,
                zero_init_last=(goal_mode == "adaln"),
            )
        if goal_mode == "channel":
            self.goal_channel_proj = zero_init_linear(nn.Linear(embed_dim, embed_dim))
        if goal_mode == "cross":
            self.goal_cross_proj = zero_init_linear(nn.Linear(embed_dim, embed_dim))
            self.goal_type_embed = nn.Parameter(torch.zeros(embed_dim))

        if config.grpo:
            self._goal_defer_grpo = False
            self._init_grpo(config.grpo_cfg)

    # ------------------------------------------------------------------ goal plumbing

    def _init_grpo(self, cfg):
        """Deferred and tolerant version of the base GRPO setup.

        Deferred because the base class calls this from ``__init__`` before the
        goal modules exist, which would otherwise make ``old_policy`` a
        goal-blind copy.  Tolerant because a goal-conditioned planner is normally
        warm-started from a goal-free IL checkpoint, so the goal parameters are
        legitimately absent and ``strict=True`` would raise.
        """
        if getattr(self, "_goal_defer_grpo", False):
            return

        from pathlib import Path

        from navsim.common.dataloader import MetricCacheLoader
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
            PDMSimulator,
        )
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        self.denoised_clip_value = cfg.denoised_clip_value
        self.eval_randn_clip_value = cfg.eval_randn_clip_value
        self.randn_clip_value = cfg.randn_clip_value
        self.final_action_clip_value = cfg.final_action_clip_value
        self.eps_clip_value = cfg.eps_clip_value
        self.eval_min_sampling_denoising_std = cfg.eval_min_sampling_denoising_std
        self.min_sampling_denoising_std = cfg.min_sampling_denoising_std
        self.min_logprob_denoising_std = cfg.min_logprob_denoising_std
        self.clip_advantage_lower_quantile = cfg.clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = cfg.clip_advantage_upper_quantile
        self.gamma_denoising = cfg.gamma_denoising

        self.metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        self.simulator = PDMSimulator(proposal_sampling)
        self.train_scorer = PDMScorer(proposal_sampling, cfg.scorer_config)

        try:
            load_kw = {"map_location": "cpu"}
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kw["weights_only"] = False
            state_dict = torch.load(cfg.reference_policy_checkpoint, **load_kw)["state_dict"]
            model_dict = self.state_dict()
            filtered_ckpt = {}
            for k, v in state_dict.items():
                k2 = k[len("agent.action_head."):] if k.startswith("agent.action_head.") else k
                if k2 in model_dict and v.shape == model_dict[k2].shape:
                    filtered_ckpt[k2] = v
            missing, unexpected = self.load_state_dict(filtered_ckpt, strict=False)
            goal_missing = [k for k in missing if k.startswith("goal_")]
            print(
                f"[GoalCondDiffusionPlanner] Loaded {len(filtered_ckpt)} tensors from "
                f"{cfg.reference_policy_checkpoint}. Missing: {len(missing)} "
                f"({len(goal_missing)} goal params, expected when warm-starting from a "
                f"goal-free checkpoint), Unexpected: {len(unexpected)}."
            )
        except FileNotFoundError:
            print(f"Warning: GRPO checkpoint not found at {cfg.reference_policy_checkpoint}. Skipping loading.")

        self.old_policy = copy.deepcopy(self)
        self.old_policy.eval()
        for param in self.old_policy.parameters():
            param.requires_grad = False

    @contextmanager
    def goal_context(self, goal: Optional[torch.Tensor]):
        """Temporarily bind a goal so nested calls pick it up without new arguments."""
        previous = getattr(self, "_goal_ctx", None)
        self._goal_ctx = goal
        try:
            yield self
        finally:
            self._goal_ctx = previous

    def _resolve_goal(self, goal: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
        """Return the goal expanded to ``batch_size``, or ``None`` if inactive.

        Both GRPO (``repeat_interleave(G, 0)``) and ``get_logprobs``
        (``unsqueeze(1).repeat(1, steps, ...).flatten(0, 1)``) replicate each
        sample contiguously, so a single ``repeat_interleave`` by the batch ratio
        reproduces their ordering, including when the two are nested.
        """
        resolved = goal if goal is not None else getattr(self, "_goal_ctx", None)
        if resolved is None or self.goal_mode == "none":
            return None
        if resolved.shape[0] == batch_size:
            return resolved
        if batch_size % resolved.shape[0] == 0:
            return resolved.repeat_interleave(batch_size // resolved.shape[0], dim=0)
        raise ValueError(
            f"Cannot broadcast goal with batch {resolved.shape[0]} to batch {batch_size}"
        )

    def encode_goal(self, goal: torch.Tensor) -> torch.Tensor:
        """Encode a goal with optional *disjoint* mask/noise corruption.

        When ``goal_noise_p > 0`` the training batch is partitioned into three
        mutually-exclusive cases by one random draw per sample:

        * ``u < goal_dropout_p``: masked goal (zero embedding);
        * ``goal_dropout_p <= u < goal_dropout_p + goal_noise_p``: noisy goal;
        * otherwise: clean GT goal.

        This gives the intended robust-teacher recipe directly, e.g.
        dropout=0.10 + noise=0.20 -> 10% masked / 20% noisy / 70% clean.
        With ``goal_noise_p == 0`` this is backward compatible with the old
        independent goal-dropout behaviour.
        """
        goal_eff = goal[..., :3]
        drop_mask = None
        noise_mask = None
        if self.training and (self.goal_dropout_p > 0.0 or self.goal_noise_p > 0.0):
            u = torch.rand(goal_eff.shape[0], device=goal_eff.device)
            drop_mask = u < self.goal_dropout_p
            noise_mask = (u >= self.goal_dropout_p) & (u < self.goal_dropout_p + self.goal_noise_p)

            # Vectorised corruption avoids per-batch GPU synchronisation from
            # ``mask.any().item()`` / dynamic-size random tensors.
            if self.goal_noise_std_xy > 0.0 or self.goal_noise_std_heading > 0.0:
                goal_eff = goal_eff.clone()
                mask_f = noise_mask[:, None].to(goal_eff.dtype)
                if self.goal_noise_std_xy > 0.0:
                    goal_eff[:, :2] += (
                        torch.randn_like(goal_eff[:, :2])
                        * self.goal_noise_std_xy
                        * mask_f
                    )
                if self.goal_noise_std_heading > 0.0:
                    goal_eff[:, 2:3] += (
                        torch.randn_like(goal_eff[:, 2:3])
                        * self.goal_noise_std_heading
                        * mask_f
                    )

        goal_norm = self.norm_odo(goal_eff.unsqueeze(1)).squeeze(1)
        embedding = self.goal_encoder(goal_norm)
        if drop_mask is not None:
            embedding = embedding.masked_fill(drop_mask[:, None], 0.0)
        return embedding

    def _inpaint_endpoint(self, x_recon: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Replace the endpoint of the predicted clean trajectory with the goal."""
        goal_norm = self.norm_odo(goal[..., :3].unsqueeze(1)).squeeze(1).to(x_recon.dtype)
        weight = self.goal_inpaint_weight
        last = x_recon[:, -1, :]
        new_xy = (1.0 - weight) * last[..., :2] + weight * goal_norm[..., :2]
        if self.goal_inpaint_heading:
            new_heading = (1.0 - weight) * last[..., 2:3] + weight * goal_norm[..., 2:3]
        else:
            new_heading = last[..., 2:3]
        new_last = torch.cat([new_xy, new_heading], dim=-1).unsqueeze(1)
        return torch.cat([x_recon[:, :-1, :], new_last], dim=1)

    def _dit_step(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        vl_embeds: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        goal_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fuse the inputs, inject the goal, run the DiT, decode. Returns ``(B, H, action_dim)``."""
        action_features = self.action_encoder(actions, timesteps)
        if hasattr(self, "position_embedding"):
            pos_ids = torch.arange(action_features.shape[1], device=actions.device)
            action_features = action_features + self.position_embedding(pos_ids)

        vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
        fused_input = self.fusion_projector(
            torch.cat((his_traj_features, vl_embeds_mean, action_features), dim=2)
        )

        encoder_states = vl_embeds
        conditioning = ego_status_features

        if goal_embedding is not None:
            if self.goal_mode == "adaln":
                conditioning = conditioning + goal_embedding
            elif self.goal_mode == "channel":
                fused_input = fused_input + self.goal_channel_proj(goal_embedding).unsqueeze(1)
            elif self.goal_mode == "cross":
                goal_token = self.goal_cross_proj(goal_embedding) + self.goal_type_embed
                encoder_states = torch.cat(
                    [vl_embeds, goal_token.unsqueeze(1).to(vl_embeds.dtype)], dim=1
                )

        model_output = self.model(
            hidden_states=fused_input,
            encoder_hidden_states=encoder_states,
            conditioning_features=conditioning,
            timesteps=timesteps,
        )
        return self.action_decoder(model_output)

    def _goal_embedding_for(self, goal: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
        if goal is None or self.goal_mode not in TRAINABLE_GOAL_MODES:
            return None
        return self.encode_goal(goal).to(dtype)

    # ------------------------------------------------------------------ overrides

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
        """Base implementation with goal injection and the inpainting hook added."""
        resolved_goal = self._resolve_goal(goal, x.shape[0])

        model_dtype = next(self.model.parameters()).dtype
        x = x.to(model_dtype)

        goal_embedding = self._goal_embedding_for(resolved_goal, model_dtype)
        pred_noise = self._dit_step(
            x, t, vl_features, his_traj_features, ego_status_features, goal_embedding
        )

        if self.config.sampling_method == "ddpm":
            x_recon = self.extract(self.ddpm_sqrt_recip_alphas_cumprod, t, x.shape) * x - \
                      self.extract(self.ddpm_sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise
        elif self.config.sampling_method == "ddim":
            alpha_t = self.extract(self.ddim_alphas, index, x.shape)
            sqrt_one_minus_alpha_t = self.extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
            x_recon = (x - sqrt_one_minus_alpha_t * pred_noise) / (alpha_t ** 0.5)
        else:
            raise ValueError(f"p_mean_variance not supported for method: {self.config.sampling_method}")

        denoised_clip_value = getattr(self, "denoised_clip_value", 1.0)
        x_recon = x_recon.clamp(-denoised_clip_value, denoised_clip_value)

        # Reconstruction guidance: pin the endpoint to the privileged goal. Placed
        # after the clamp and before the mean, so the DDIM branch below rederives
        # pred_noise from the modified x_recon and the guidance propagates properly.
        if resolved_goal is not None and self.goal_mode == "inpaint":
            x_recon = self._inpaint_endpoint(x_recon, resolved_goal)

        if self.config.sampling_method == "ddpm":
            model_mean = self.extract(self.ddpm_mu_coef1, t, x.shape) * x_recon + \
                         self.extract(self.ddpm_mu_coef2, t, x.shape) * x
            model_log_variance = self.extract(self.ddpm_logvar_clipped, t, x.shape)
        else:
            alpha_prev = self.extract(self.ddim_alphas_prev, index, x.shape)

            pred_noise = (x - (alpha_t ** 0.5) * x_recon) / sqrt_one_minus_alpha_t

            eps_clip_value = getattr(self, "eps_clip_value", None)
            if eps_clip_value is not None:
                pred_noise = pred_noise.clamp(-eps_clip_value, eps_clip_value)

            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(x).unsqueeze(1)

            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) ** 0.5
            ).clamp(min=1e-10)

            pred_dir_xt = (1.0 - alpha_prev - sigma ** 2).clamp(min=0).sqrt() * pred_noise
            model_mean = (alpha_prev ** 0.5) * x_recon + pred_dir_xt
            model_log_variance = torch.log(sigma ** 2 + 1e-20)

        return model_mean, model_log_variance, x_recon

    def forward(self, vl_features: torch.Tensor, action_input: BatchFeature) -> BatchFeature:
        """Training loss, with the goal injected into the denoiser."""
        goal = self._resolve_goal(extract_goal(action_input), vl_features.shape[0])

        vl_embeds = self.feature_encoder(vl_features)
        his_traj_features = self.his_traj_encoder(
            action_input.his_traj.unsqueeze(1)
        ).repeat(1, self.config.action_horizon, 1)
        ego_status_features = self.ego_status_encoder(action_input.status_feature)

        gt_actions = self.norm_odo(action_input.action)
        goal_embedding = self._goal_embedding_for(goal, gt_actions.dtype)

        if self.config.sampling_method == "flow":
            noise = torch.randn_like(gt_actions)
            t_cont = self.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)
            t_cont_reshaped = t_cont[:, None, None]

            noisy_actions = (1 - t_cont_reshaped) * noise + t_cont_reshaped * gt_actions
            velocity_target = gt_actions - noise
            t_discrete = (t_cont * self.num_timestep_buckets).long()

            pred_velocity = self._dit_step(
                noisy_actions, t_discrete, vl_embeds, his_traj_features,
                ego_status_features, goal_embedding,
            )
            loss = torch.nn.functional.mse_loss(pred_velocity, velocity_target, reduction="mean")
        else:
            noise = torch.randn_like(gt_actions)
            t_discrete = self.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)

            noisy_actions = (
                self.extract(self.ddpm_sqrt_alphas_cumprod, t_discrete, gt_actions.shape) * gt_actions +
                self.extract(self.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, gt_actions.shape) * noise
            )

            pred_noise = self._dit_step(
                noisy_actions, t_discrete, vl_embeds, his_traj_features,
                ego_status_features, goal_embedding,
            )
            loss = torch.nn.functional.mse_loss(pred_noise, noise, reduction="mean")

        return BatchFeature(data={"loss": loss})

    def get_action(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> BatchFeature:
        """Inference. The ddpm/ddim samplers reach the goal through ``p_mean_variance``."""
        goal = extract_goal(action_input)
        with self.goal_context(goal if goal is not None else getattr(self, "_goal_ctx", None)):
            return super().get_action(vl_features, action_input, init_actions, deterministic)

    def forward_grpo(self, vl_features: torch.Tensor, action_input: BatchFeature, tokens_list, **kwargs):
        """GRPO, with the goal bound on both the policy and the frozen reference."""
        goal = extract_goal(action_input)
        if goal is None:
            goal = getattr(self, "_goal_ctx", None)

        with self.goal_context(goal):
            if getattr(self, "old_policy", None) is not None:
                with self.old_policy.goal_context(goal):
                    return super().forward_grpo(vl_features, action_input, tokens_list, **kwargs)
            return super().forward_grpo(vl_features, action_input, tokens_list, **kwargs)
