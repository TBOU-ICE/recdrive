from typing import Any

from .recogdrive_agent import ReCogDriveAgent
from .recogdrive_dit_temporal_multi_teacher_distill_trainer import ReCogDriveTemporalMultiTeacherDiTDistillTrainer


class ReCogDriveTemporalMultiTeacherDiTDistillAgent(ReCogDriveAgent):
    """Temporal multi-teacher DiT distillation agent.

    Extends ReCogDriveAgent without modifying it.  The parent builds both frozen
    teacher DiTs (IL + RL) and the trainable student.  This subclass replaces the
    standard ReCogDriveDiTDistillTrainer with
    ReCogDriveTemporalMultiTeacherDiTDistillTrainer, which adds an EC-style
    temporal consistency loss on top of the dual-teacher KL distillation.

    The parent's forward() dispatches to self.dit_distill_trainer.compute_loss(
        student, teacher_il, teacher_rl, vl_features, action_input)
    — this matches the new trainer's signature exactly.
    """

    def __init__(
        self,
        *args: Any,
        dit_temporal_loss_weight: float = 0.05,
        dit_temporal_shift_steps: int = 1,
        dit_temporal_pos_weight: float = 1.0,
        dit_temporal_heading_weight: float = 0.2,
        dit_temporal_acc_weight: float = 0.1,
        dit_temporal_jerk_weight: float = 0.05,
        dit_temporal_yaw_rate_weight: float = 0.1,
        dit_temporal_yaw_acc_weight: float = 0.05,
        dit_temporal_dt: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not getattr(self, "dit_distill", False):
            raise ValueError(
                "ReCogDriveTemporalMultiTeacherDiTDistillAgent requires dit_distill=True."
            )
        if self.teacher_il_action_head is None or self.teacher_rl_action_head is None:
            raise RuntimeError(
                "Temporal multi-teacher agent requires both teacher_il_action_head "
                "and teacher_rl_action_head. Make sure teacher_dit_checkpoint_il and "
                "teacher_dit_checkpoint_rl are set."
            )

        # Replace the standard dual-teacher trainer built by the parent with the
        # temporal-consistency version.
        self.dit_distill_trainer = ReCogDriveTemporalMultiTeacherDiTDistillTrainer(
            eps_clip=float(kwargs.get("dit_distill_eps_clip", 0.2)),
            min_sigma=float(kwargs.get("dit_distill_min_sigma", 0.04)),
            normalize_advantage=bool(kwargs.get("dit_distill_normalize_advantage", True)),
            log_dir=kwargs.get("dit_distill_log_dir", None) or None,
            log_interval=int(kwargs.get("dit_distill_log_interval", 50)),
            il_weight=float(kwargs.get("dit_distill_il_weight", 0.75)),
            rl_weight=float(kwargs.get("dit_distill_rl_weight", 0.25)),
            temporal_loss_weight=dit_temporal_loss_weight,
            temporal_shift_steps=dit_temporal_shift_steps,
            temporal_pos_weight=dit_temporal_pos_weight,
            temporal_heading_weight=dit_temporal_heading_weight,
            temporal_acc_weight=dit_temporal_acc_weight,
            temporal_jerk_weight=dit_temporal_jerk_weight,
            temporal_yaw_rate_weight=dit_temporal_yaw_rate_weight,
            temporal_yaw_acc_weight=dit_temporal_yaw_acc_weight,
            temporal_dt=dit_temporal_dt,
        )
