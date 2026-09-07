"""Goal-point conditioning utilities (privileged information).

This module is new and is not imported by any pre-existing training or
evaluation script, so adding it cannot change the behaviour of current runs.

The goal point is the endpoint of the ground-truth trajectory, i.e.
``targets["trajectory"][:, -1, :]`` with layout ``(x, y, heading)``.  It is the
same quantity GoalFlow feeds to its teacher (``gt_trajs[:, 7:8, :2]`` in
``goalflow_model_traj.py``), which uses only ``(x, y)`` and no clustering --
the 8192-entry vocabulary there exists solely so the deployable student can
*predict* the goal, which a privileged teacher never has to do.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# "none"    : disabled, planner behaves exactly like the base class
# "inpaint" : training-free diagnostic, overwrites the predicted endpoint
# "adaln"   : goal is added to the AdaLN conditioning vector (with ego status)
# "channel" : goal is added into the residual stream alongside the fused input
# "cross"   : goal is appended to the cross-attention key/value sequence
GOAL_MODES = ("none", "inpaint", "adaln", "channel", "cross")

TRAINABLE_GOAL_MODES = ("adaln", "channel", "cross")


def pos2posemb2d(pos: torch.Tensor, num_pos_feats: int = 128, temperature: float = 10000.0) -> torch.Tensor:
    """Sine/cosine embedding of a 2D position, mirroring GoalFlow's ``utils.pos2posemb2d``.

    ``pos`` is expected to be *normalised* to roughly ``[-1, 1]`` rather than raw
    metres.  GoalFlow feeds raw metres into this function even though the
    ``pos * 2 * pi`` scaling and ``temperature=10000`` are designed for unit-range
    inputs, which wastes most of the frequency bands; normalising first keeps the
    embedding well conditioned.

    :param pos: tensor of shape ``(..., >=2)``, only the first two channels are used
    :param num_pos_feats: number of features per coordinate, must be even
    :return: tensor of shape ``(..., 2 * num_pos_feats)``
    """
    if num_pos_feats % 2 != 0:
        raise ValueError(f"num_pos_feats must be even, got {num_pos_feats}")

    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.floor(dim_t / 2) / num_pos_feats)

    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


class GoalEncoder(nn.Module):
    """Encodes a normalised goal point into a conditioning vector.

    ``zero_init_last`` controls whether the final linear layer starts at zero.
    The goal branch as a whole must contain EXACTLY ONE zero-initialised layer,
    and it must be the OUTERMOST one (the ControlNet zero-conv rule): a single
    zero layer makes the branch a no-op at init -- so a goal-conditioned planner
    warm-started from a goal-free checkpoint reproduces that checkpoint exactly
    until the first gradient step -- while still receiving gradient itself.
    Stacking TWO zero layers (e.g. a zero-init encoder followed by a zero-init
    projection) deadlocks instead: with ``y = P e``, ``dL/dP = g e^T = 0``
    because ``e = 0``, and ``dL/de = P^T g = 0`` because ``P = 0``, so neither
    layer ever updates and the goal can never influence the output.

    Therefore set ``zero_init_last=True`` only when this encoder's output is
    consumed directly (adaln mode); set it ``False`` when a separate
    zero-initialised projection follows (channel / cross modes).
    """

    def __init__(
        self,
        out_dim: int,
        hidden_dim: int = 1024,
        sincos_dim: int = 128,
        use_heading: bool = False,
        zero_init_last: bool = True,
    ):
        super().__init__()
        self.sincos_dim = sincos_dim
        self.use_heading = use_heading

        in_dim = 2 * sincos_dim + (2 if use_heading else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        if zero_init_last:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, goal_norm: torch.Tensor) -> torch.Tensor:
        """:param goal_norm: ``(B, >=2)`` goal normalised to ``[-1, 1]`` by ``norm_odo``."""
        embedding = pos2posemb2d(goal_norm[..., :2], self.sincos_dim).to(goal_norm.dtype)
        if self.use_heading:
            heading = goal_norm[..., 2:3] * math.pi
            embedding = torch.cat([embedding, torch.sin(heading), torch.cos(heading)], dim=-1)
        return self.net(embedding)


def zero_init_linear(layer: nn.Linear) -> nn.Linear:
    """Zero out a linear layer so the branch it feeds starts as a no-op."""
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer
