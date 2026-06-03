from typing import Any

from .recogdrive_agent import ReCogDriveAgent
from .recogdrive_dit_temporal_distill_trainer import ReCogDriveTemporalDiTDistillTrainer


class ReCogDriveTemporalDiTDistillAgent(ReCogDriveAgent):
    """Non-invasive temporal DiT-OPD agent.

    This class keeps the original ReCogDriveAgent / DiT-OPD code untouched.  It
    builds the same frozen teacher DiT and trainable student DiT through
    ReCogDriveAgent, then replaces only the DiT distillation trainer with a
    temporal-consistency version.
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
            raise ValueError("ReCogDriveTemporalDiTDistillAgent requires dit_distill=True.")
        if self.teacher_action_head is None:
            raise RuntimeError("Temporal DiT distill agent did not build teacher_action_head.")

        self.dit_distill_trainer = ReCogDriveTemporalDiTDistillTrainer(
            eps_clip=float(kwargs.get("dit_distill_eps_clip", 0.2)),
            min_sigma=float(kwargs.get("dit_distill_min_sigma", 0.04)),
            normalize_advantage=bool(kwargs.get("dit_distill_normalize_advantage", True)),
            log_dir=kwargs.get("dit_distill_log_dir", None) or None,
            log_interval=int(kwargs.get("dit_distill_log_interval", 50)),
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
