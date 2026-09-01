"""GoalBridge v2 agent: dense local OPD + trajectory-support JSD."""
from __future__ import annotations

from .recogdrive_goalbridge_agent import ReCogDriveGoalBridgeAgent
from .recogdrive_distributional_goalbridge_distill_trainer import (
    ReCogDriveDistributionalGoalBridgeDistillTrainer,
)


class ReCogDriveDistributionalGoalBridgeAgent(ReCogDriveGoalBridgeAgent):
    def __init__(
        self,
        *args,
        support_weight: float = 0.25,
        support_teacher_candidates: int = 4,
        support_student_candidates: int = 4,
        support_temperature: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        base = self.scene_router_trainer
        self.scene_router_trainer = ReCogDriveDistributionalGoalBridgeDistillTrainer(
            bucket_names=base.bucket_names,
            fallback_bucket=base.fallback_bucket,
            min_sigma=base.min_sigma,
            smooth_weight=base.smooth_weight,
            goal_loss_weight=base.goal_loss_weight,
            kd_weight=base.kd_weight,
            anchor_weight=base.anchor_weight,
            recoverability_tau_m=base.recoverability_tau_m,
            recoverability_floor=base.recoverability_floor,
            kl_precision_clip=base.kl_precision_clip,
            collect_viz=base.collect_viz,
            kd_gate_mode=base.kd_gate_mode,
            anchor_gate_mode=base.anchor_gate_mode,
            support_weight=support_weight,
            support_teacher_candidates=support_teacher_candidates,
            support_student_candidates=support_student_candidates,
            support_temperature=support_temperature,
        )
        print(
            "[GoalBridge-v2] trajectory-support distillation enabled: "
            f"support_w={support_weight:g}, Kt={support_teacher_candidates}, "
            f"Ks={support_student_candidates}, T={support_temperature:g}"
        )
