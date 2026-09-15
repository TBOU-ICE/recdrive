"""Deployable goal-conditioned planner whose goal is predicted from observations.

The privileged teacher receives the ground-truth endpoint.  This student never
receives that endpoint at inference: a lightweight goal head predicts a
normalised endpoint from the same VLM / history / ego features already consumed
by the DiT, then the inherited GoalCondDiffusionPlanner uses the predicted goal.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .recogdrive_goal_planner import GoalCondDiffusionPlanner
from .recogdrive_diffusion_planner import ReCogDriveDiffusionPlannerConfig


class GoalPredictionHead(nn.Module):
    """Predict a single normalised (x, y, heading) goal from encoded observations."""

    def __init__(self, embed_dim: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 3),
        )
        # Start near the centre of the trajectory normalisation range rather
        # than with a large arbitrary endpoint.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        vl_embeds: torch.Tensor,
        his_embeds: torch.Tensor,
        ego_embeds: torch.Tensor,
    ) -> torch.Tensor:
        vl_pool = vl_embeds.mean(dim=1)
        his_pool = his_embeds.mean(dim=1)
        fused = torch.cat([vl_pool, his_pool, ego_embeds], dim=-1)
        return torch.tanh(self.net(fused))


class PredictedGoalDiffusionPlanner(GoalCondDiffusionPlanner):
    """Goal-conditioned DiT plus a deployable observation->goal prediction head."""

    def __init__(
        self,
        config: ReCogDriveDiffusionPlannerConfig,
        goal_mode: str = "adaln",
        goal_sincos_dim: int = 128,
        goal_hidden_dim: int = 1024,
        goal_use_heading: bool = False,
        goal_predictor_hidden_dim: int = 512,
        goal_predictor_dropout: float = 0.0,
        goal_dropout_p: float = 0.0,
        goal_noise_p: float = 0.0,
        goal_noise_std_xy: float = 0.0,
        goal_noise_std_heading: float = 0.0,
    ):
        # Goal corruption stays off by default so OPD behaviour is unchanged.
        # Teacher SFT turns it on: conditioning only ever on goals that are
        # exactly consistent with the target trajectory teaches the DiT to treat
        # the goal as a hard constraint, which collapses the moment OPD starts
        # feeding it a predicted goal that carries real error.
        super().__init__(
            config,
            goal_mode=goal_mode,
            goal_sincos_dim=goal_sincos_dim,
            goal_hidden_dim=goal_hidden_dim,
            goal_use_heading=goal_use_heading,
            goal_dropout_p=goal_dropout_p,
            goal_noise_p=goal_noise_p,
            goal_noise_std_xy=goal_noise_std_xy,
            goal_noise_std_heading=goal_noise_std_heading,
        )
        self.goal_predictor = GoalPredictionHead(
            config.input_embedding_dim,
            hidden_dim=goal_predictor_hidden_dim,
            dropout=goal_predictor_dropout,
        )

    def predict_goal_from_encoded(
        self,
        vl_embeds: torch.Tensor,
        his_embeds: torch.Tensor,
        ego_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(goal_raw_metres, goal_norm)`` for already encoded features."""
        # Keep the auxiliary goal objective from reshaping the pretrained scene
        # encoders directly; KD still updates those encoders through the normal
        # DiT path, while the goal head learns on a stable detached representation.
        goal_norm = self.goal_predictor(vl_embeds.detach(), his_embeds.detach(), ego_embeds.detach())
        goal_raw = self.denorm_odo(goal_norm.unsqueeze(1)).squeeze(1)
        return goal_raw, goal_norm

    def predict_goal(
        self,
        vl_features: torch.Tensor,
        his_traj: torch.Tensor,
        ego_status: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = next(self.parameters()).dtype
        vl_embeds = self.feature_encoder(vl_features.to(dtype))
        his_embeds = self.his_traj_encoder(his_traj.to(dtype).unsqueeze(1)).repeat(
            1, self.config.action_horizon, 1
        )
        ego_embeds = self.ego_status_encoder(ego_status.to(dtype))
        return self.predict_goal_from_encoded(vl_embeds, his_embeds, ego_embeds)
