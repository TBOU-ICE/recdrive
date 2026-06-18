from __future__ import annotations

import inspect
import os
import sys
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf


_DEFAULT_DRIVEVLA_ROOT = "/workspace/volumes/ad-e2e-al-sh01/nby/HUGSIM_DriveMem_base"
_DEFAULT_DRIVEVLA_CONFIG = os.path.join(_DEFAULT_DRIVEVLA_ROOT, "episode_drive.yaml")


def _ensure_episode_drive_on_path(root: str) -> None:
    if root not in sys.path:
        sys.path.insert(0, root)


class DriveVLAM0Teacher(torch.nn.Module):
    """Frozen DriveVLA-M0 score-head evaluator for student trajectory candidates."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        root_dir: str = _DEFAULT_DRIVEVLA_ROOT,
    ) -> None:
        super().__init__()
        _ensure_episode_drive_on_path(root_dir)

        from EpisodeDrive.action_head import ActionHead  # pylint: disable=import-error

        config_path = config_path or _DEFAULT_DRIVEVLA_CONFIG
        cfg = OmegaConf.load(config_path)
        self._config = cfg.action_head_config
        self.action_head = ActionHead(self._config).cuda()

        load_kw: Dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kw["weights_only"] = False
        ckpt = torch.load(checkpoint_path, **load_kw)
        state = ckpt.get("state_dict", ckpt)

        stripped = {}
        for k, v in state.items():
            if k.startswith("agent.action_head."):
                stripped[k[len("agent.action_head."):]] = v
            elif k.startswith("action_head."):
                stripped[k[len("action_head."):]] = v

        missing, unexpected = self.action_head.load_state_dict(stripped, strict=False)
        print(
            "[DriveVLA teacher] Loaded action head from "
            f"{checkpoint_path}. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

        for p in self.action_head.parameters():
            p.requires_grad = False
        self.action_head.eval()

    def score_trajectories(
        self,
        last_hidden_state: torch.Tensor,
        status_feature: torch.Tensor,
        candidate_trajs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Score external candidate trajectories with the DriveVLA-M0 scorer.

        Args:
            last_hidden_state: Cached VLM hidden states, shape (B, N, 1536).
            status_feature: Ego status tensor, shape (B, 8).
            candidate_trajs: Candidate trajectories, shape (B, K, 8, 3).
        """
        if candidate_trajs.ndim != 4:
            raise ValueError(
                f"candidate_trajs must have shape (B, K, H, D); got {tuple(candidate_trajs.shape)}"
            )

        B, K, _, _ = candidate_trajs.shape
        device = candidate_trajs.device
        dtype = next(self.action_head.parameters()).dtype

        with torch.no_grad():
            status_feature = status_feature.to(device=device, dtype=dtype)
            last_hidden_state = last_hidden_state.to(device=device, dtype=dtype)
            proposals = candidate_trajs.to(device=device, dtype=dtype).detach()

            ego_status = torch.cat(
                [torch.zeros_like(status_feature)[:, :3], status_feature],
                dim=1,
            )
            ego_token = self.action_head.hist_encoding(ego_status)[:, None]
            scene_features = self.action_head.q_former(
                self.action_head.scene_embeds, last_hidden_state
            )

            embedded_traj = self.action_head.pos_embed(proposals.reshape(B, K, -1))
            tr_out = self.action_head.scorer_attention(embedded_traj, scene_features)
            tr_out = tr_out + ego_token
            pred_logit, *_ = self.action_head.scorer(proposals, tr_out)

            pdm_score = (
                self._config.noc * pred_logit["no_at_fault_collisions"].sigmoid().log()
                + self._config.dac * pred_logit["drivable_area_compliance"].sigmoid().log()
                + self._config.ddc * pred_logit["driving_direction_compliance"].sigmoid().log()
                + (
                    self._config.ttc * pred_logit["time_to_collision_within_bound"].sigmoid()
                    + self._config.ep * pred_logit["ego_progress"].sigmoid()
                    + self._config.comfort * pred_logit["comfort"].sigmoid()
                ).log()
            )

        return {
            "pdm_score": pdm_score,
            "pred_logit": pred_logit,
        }
