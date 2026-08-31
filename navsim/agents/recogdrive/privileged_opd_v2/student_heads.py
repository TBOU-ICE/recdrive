"""Training-only student heads for Privileged-OPD v2."""
from __future__ import annotations

import math
import torch
from torch import nn

from navsim.agents.recogdrive.goal_cond import pos2posemb2d


class AuxiliaryGoalHead(nn.Module):
    """Predict the final privileged goal from observable student features.

    The prediction is an auxiliary representation-learning objective only; it is
    never fed into the deployment planner in v2-B.
    """
    def __init__(self, dim: int, hidden_dim: int = 512, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, vl_embeds: torch.Tensor, his_embeds: torch.Tensor, ego_embeds: torch.Tensor):
        vl = vl_embeds.mean(dim=1)
        his = his_embeds[:, 0] if his_embeds.ndim == 3 else his_embeds
        x = torch.cat([vl, his, ego_embeds], dim=-1)
        return self.net(x)


class GoalPreferenceHead(nn.Module):
    """Score a fixed vocabulary of candidate goal endpoints from observation."""
    def __init__(self, dim: int, hidden_dim: int = 512, sincos_dim: int = 128):
        super().__init__()
        self.sincos_dim = int(sincos_dim)
        self.scene_proj = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim)
        )
        self.goal_proj = nn.Sequential(
            nn.Linear(2 * self.sincos_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim)
        )
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        vl_embeds: torch.Tensor,
        his_embeds: torch.Tensor,
        ego_embeds: torch.Tensor,
        candidate_goal_norm: torch.Tensor,
    ) -> torch.Tensor:
        vl = vl_embeds.mean(dim=1)
        his = his_embeds[:, 0] if his_embeds.ndim == 3 else his_embeds
        scene = self.scene_proj(torch.cat([vl, his, ego_embeds], dim=-1))
        scene = torch.nn.functional.normalize(scene.float(), dim=-1)
        pos = pos2posemb2d(candidate_goal_norm[..., :2].float(), self.sincos_dim)
        goal = torch.nn.functional.normalize(self.goal_proj(pos), dim=-1)
        scale = self.logit_scale.exp().clamp(max=20.0)
        return scale * torch.einsum("bd,kd->bk", scene, goal)
