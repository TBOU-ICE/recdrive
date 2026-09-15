"""Teacher-rollout SFT agent for the deployable GoalBridge student. Additive file.

Phase B of the teacher-rollout pipeline.  This agent builds exactly the same
``PredictedGoalDiffusionPlanner`` the GoalBridge OPD agent builds, so the
checkpoint it writes drops straight into OPD's ``agent.checkpoint_path``:
``_candidate_state_keys`` maps ``agent.action_head.*`` onto ``action_head.*``,
which carries ``goal_encoder`` and ``goal_predictor`` across with the DiT.

Unlike the OPD agent it loads no teachers -- the teachers already ran offline in
Phase A -- so a single planner is resident and training is as cheap as plain IL.

Goal corruption is ON here (the robust-teacher recipe: 10% masked / 20% noisy /
70% clean, sigma_xy = 1m).  Without it ``goal_encoder`` would only ever meet
goals perfectly consistent with their trajectory, then face an imperfect
predicted goal the moment OPD starts.
"""

from __future__ import annotations

from typing import Dict

import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent, make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_predicted_goal_planner import (
    PredictedGoalDiffusionPlanner,
)
from navsim.agents.recogdrive.recogdrive_teacher_sft_trainer import (
    ReCogDriveTeacherSFTTrainer,
)


class ReCogDriveTeacherSFTAgent(ReCogDriveAgent):
    """Predicted-goal student fitted to cached privileged-teacher rollouts."""

    def __init__(
        self,
        *args,
        student_goal_mode: str = "adaln",
        goal_sincos_dim: int = 128,
        goal_hidden_dim: int = 1024,
        goal_use_heading: bool = False,
        goal_predictor_hidden_dim: int = 512,
        goal_predictor_dropout: float = 0.0,
        goal_dropout_p: float = 0.10,
        goal_noise_p: float = 0.20,
        goal_noise_std_xy: float = 1.0,
        goal_noise_std_heading: float = 0.10,
        goal_loss_weight: float = 1.0,
        traj_eval_interval: int = 50,
        sft_epochs: int = 10,
        sft_warmup_epochs: int = 1,
        **kwargs,
    ):
        kwargs["dit_distill"] = False
        kwargs["opd"] = False
        kwargs["grpo"] = False
        super().__init__(*args, **kwargs)

        old_student = self.action_head
        cfg = make_recogdrive_config(
            self.dit_type,
            action_dim=3,
            action_horizon=8,
            grpo=False,
            input_embedding_dim=384 if self.dit_type == "small" else 1536,
            sampling_method=old_student.config.sampling_method,
        )
        cfg.vlm_size = self.vlm_size
        planner = PredictedGoalDiffusionPlanner(
            cfg,
            goal_mode=student_goal_mode,
            goal_sincos_dim=goal_sincos_dim,
            goal_hidden_dim=goal_hidden_dim,
            goal_use_heading=goal_use_heading,
            goal_predictor_hidden_dim=goal_predictor_hidden_dim,
            goal_predictor_dropout=goal_predictor_dropout,
            goal_dropout_p=goal_dropout_p,
            goal_noise_p=goal_noise_p,
            goal_noise_std_xy=goal_noise_std_xy,
            goal_noise_std_heading=goal_noise_std_heading,
        ).cuda()
        # Carry over whatever the base agent already built; initialize() then
        # reloads checkpoint_path over the top.
        planner.load_state_dict(old_student.state_dict(), strict=False)
        self.action_head = planner
        for param in self.action_head.parameters():
            param.requires_grad = True

        self.teacher_sft_trainer = ReCogDriveTeacherSFTTrainer(
            goal_loss_weight=goal_loss_weight,
            traj_eval_interval=traj_eval_interval,
        )
        self._sft_epochs = int(sft_epochs)
        self._sft_warmup_epochs = int(sft_warmup_epochs)

        print(
            "[TeacherSFT] student=PredictedGoalDiffusionPlanner "
            f"mode={student_goal_mode}, goal_w={goal_loss_weight:g}; goal corruption "
            f"clean={1 - goal_dropout_p - goal_noise_p:.2f} noisy={goal_noise_p:.2f} "
            f"masked={goal_dropout_p:.2f} noise_xy={goal_noise_std_xy:g}m; "
            "target=cached teacher rollout, condition=teacher endpoint (no GT goal)"
        )

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None):
        del tokens_list  # routing already happened offline in Phase A
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()
        if not self.cache_hidden_state:
            raise RuntimeError("Teacher-rollout SFT expects cache_hidden_state=True.")

        model_dtype = next(self.action_head.parameters()).dtype
        history_trajectory = features["history_trajectory"].cuda()
        status_feature = features["status_feature"].cuda()
        last_hidden_state = features["last_hidden_state"].cuda()
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        last_hidden_state = last_hidden_state.to(model_dtype)
        history_flat = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_flat], dim=1)

        if self.training:
            if targets is None or "teacher_trajectory" not in targets:
                raise RuntimeError(
                    "Teacher-rollout SFT requires a 'teacher_trajectory' target. Run Phase A "
                    "(scripts/generation/run_teacher_rollout_cache.py) and point the training "
                    "script at its teacher_rollout.pt."
                )
            data = {
                "state": input_state.to(model_dtype),
                "his_traj": history_flat.to(model_dtype),
                "status_feature": status_feature.to(model_dtype),
                # The regression target is the teacher's trajectory, never GT.
                "action": targets["teacher_trajectory"].cuda().to(model_dtype),
            }
            if "trajectory" in targets:
                # Diagnostics only: lets the log show teacher-vs-GT and
                # student-vs-GT next to student-vs-teacher.
                data["gt_trajectory"] = targets["trajectory"].cuda().float()
            return self.teacher_sft_trainer.compute_loss(
                student_planner=self.action_head,
                vl_features=last_hidden_state,
                action_input=BatchFeature(data=data),
            )

        # Validation / inference: fully deployable, predicted goal only.
        action_inputs = BatchFeature(data={
            "state": input_state.to(model_dtype),
            "his_traj": history_flat.to(model_dtype),
            "status_feature": status_feature.to(model_dtype),
        })
        pred_goal, _ = self.action_head.predict_goal(
            last_hidden_state, history_flat, status_feature
        )
        with self.action_head.goal_context(pred_goal):
            out = self.action_head.get_action(last_hidden_state, action_inputs)
        out["pred_goal"] = pred_goal.detach()
        return out

    def compute_loss(self, features, targets, predictions):
        """Hand the whole BatchFeature to Lightning so diagnostics get logged.

        The base implementation returns ``predictions.loss`` for non-OPD agents,
        which would drop ``goal_branch_weight_norm`` and the rollout metrics --
        precisely the numbers that say whether SFT is working.
        """
        if self.training:
            return predictions
        return super().compute_loss(features, targets, predictions)

    def get_optimizers(self):
        """Match the cosine schedule to the SFT budget.

        The inherited default hardcodes ``epochs=200``; a 10-epoch SFT run under
        that schedule would decay the LR by almost nothing.
        """
        from navsim.agents.recogdrive.utils.lr_scheduler import WarmupCosLR

        optimizers = super().get_optimizers()
        optimizer = optimizers["optimizer"] if isinstance(optimizers, dict) else optimizers
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=max(self._sft_epochs, 1),
            # WarmupCosLR.get_init_lr divides by warmup_epochs, so it must be >= 1.
            warmup_epochs=max(1, min(self._sft_warmup_epochs, max(self._sft_epochs - 1, 1))),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
