"""Additive pure OPRD multi-teacher DiT distillation agent."""

from __future__ import annotations

from typing import Any

from .recogdrive_agent import ReCogDriveAgent
from .recogdrive_dit_oprd_multi_teacher_trainer import ReCogDriveDiTOPRDMultiTeacherTrainer


class ReCogDriveOPRDMultiTeacherDiTAgent(ReCogDriveAgent):
    """ReCogDrive agent using pure hidden-state multi-teacher OPRD for DiT.

    This subclass does not edit the base ReCogDriveAgent.  The parent still
    builds the trainable student DiT and both frozen IL/RL teacher DiTs.  We only
    replace ``self.dit_distill_trainer`` with a representation-only trainer.
    """

    def __init__(
        self,
        *args: Any,
        oprd_mid_il_weight: float = 1.0,
        oprd_last_il_weight: float = 0.85,
        oprd_last_rl_weight: float = 0.15,
        oprd_final_repr_weight: float = 1.0,
        oprd_use_final_repr: bool = True,
        oprd_normalize_hidden: bool = True,
        oprd_loss_type: str = "mse",
        oprd_mid_layer_start_ratio: float = 1.0 / 3.0,
        oprd_mid_layer_end_ratio: float = 2.0 / 3.0,
        oprd_last_layer_start_ratio: float = 2.0 / 3.0,
        oprd_middle_step_start_ratio: float = 1.0 / 3.0,
        oprd_late_step_start_ratio: float = 2.0 / 3.0,
        oprd_traj_il_weight: float = 0.0,
        oprd_traj_rl_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not getattr(self, "dit_distill", False):
            raise ValueError("ReCogDriveOPRDMultiTeacherDiTAgent requires dit_distill=True.")
        if self.teacher_il_action_head is None or self.teacher_rl_action_head is None:
            raise RuntimeError(
                "Pure OPRD multi-teacher agent requires both teacher_il_action_head "
                "and teacher_rl_action_head. Set teacher_dit_checkpoint_il and "
                "teacher_dit_checkpoint_rl."
            )

        self.dit_distill_trainer = ReCogDriveDiTOPRDMultiTeacherTrainer(
            min_sigma=float(kwargs.get("dit_distill_min_sigma", 0.04)),
            log_dir=kwargs.get("dit_distill_log_dir", None) or None,
            log_interval=int(kwargs.get("dit_distill_log_interval", 50)),
            mid_il_weight=oprd_mid_il_weight,
            last_il_weight=oprd_last_il_weight,
            last_rl_weight=oprd_last_rl_weight,
            final_repr_weight=oprd_final_repr_weight,
            use_final_repr=oprd_use_final_repr,
            normalize_hidden=oprd_normalize_hidden,
            loss_type=oprd_loss_type,
            mid_layer_start_ratio=oprd_mid_layer_start_ratio,
            mid_layer_end_ratio=oprd_mid_layer_end_ratio,
            last_layer_start_ratio=oprd_last_layer_start_ratio,
            middle_step_start_ratio=oprd_middle_step_start_ratio,
            late_step_start_ratio=oprd_late_step_start_ratio,
            traj_il_weight=oprd_traj_il_weight,
            traj_rl_weight=oprd_traj_rl_weight,
        )
