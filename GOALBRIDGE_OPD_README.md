# GoalBridge-OPD v1.1 (reviewed)

This patch implements a deployable privileged-goal distillation path for ReCogDrive.

## Core path

Old privileged OPD:

`Scene -> goal-free Student`, while the routed Teacher sees `GT goal`.

GoalBridge:

`Scene -> Goal Head -> predicted goal -> goal-conditioned Student DiT`.

The routed Teacher still receives the GT endpoint. The Student never receives GT goal at inference.

## Objective

`L = goal_loss_weight * L_goal + kd_weight * c * L_reverse_KL + anchor_weight * (1-c) * L_anchor + smooth_weight * L_jerk`

- `L_goal`: SmoothL1 on normalized endpoint `(x,y)` for the current `goal_use_heading=false` setup. If heading conditioning is enabled later, the loss automatically uses all 3 dimensions.
- `L_reverse_KL`: shared-variance DDIM reverse-transition Gaussian KL mean term. Student and Teacher first predict their own `x0`; both reverse means are then reconstructed with the **same detached student DDIM sigma**. This fixes the invalid comparison between stochastic Student `mu` (eta=1) and deterministic Teacher `mu` (eta=0).
- KL denominator uses `max(sigma, min_sigma)` only as a numerical precision floor. The DDIM reverse-mean geometry uses the actual sigma, so late denoising steps remain valid.
- KL precision is clipped and divided by the fixed clip constant, preserving relative timestep weighting while keeping the effective weight in `(0,1]`.
- Recoverability gate: `c = exp(-FDE_goal^2 / (2 tau^2))`, detached. Privileged KD is multiplied by `c`.
- `L_anchor`: frozen goal-free deployable policy reverse-mean matching, multiplied by `(1-c)`. It is a safety regularizer, not a required component. Setting `ANCHOR_WEIGHT=0` now avoids loading the extra anchor planner entirely.

Defaults:

- `GOAL_LOSS_WEIGHT=1.0`
- `KD_WEIGHT=1.0`
- `ANCHOR_WEIGHT=0.15`
- `RECOVERABILITY_TAU_M=2.0`
- `RECOVERABILITY_FLOOR=0.05`
- `KL_PRECISION_CLIP=25.0`
- `LR=5e-5`

For the anchor, prefer your strongest **deployable non-privileged** checkpoint (e.g. the previous ~89-PDMS distilled student) rather than a weaker IL-only base. The script's fallback is the original IL checkpoint because its exact path is known.

## 1. Run GoalBridge OPD now with the OLD four privileged teachers

```bash
bash scripts/training/run_recogdrive_train_goalbridge_opd_8gpu.sh
```

The default teacher checkpoints are the existing privileged goal teachers.

Important logs:

- `train/goal_loss`
- `train/goal_fde_m`
- `train/recoverability_mean`
- `train/reverse_kl_loss`
- `train/anchor_loss`
- `train/student_fde_gt_m`
- `train/x0_gap_m`

Expected dynamics:

1. `goal_fde_m` falls first.
2. `recoverability_mean` rises.
3. Privileged KD automatically becomes stronger as the predicted goal becomes recoverable.
4. Anchor pressure automatically becomes weaker as recoverability rises.
5. If `goal_fde_m` remains high, improve the Goal Head before changing OPD loss again.

To switch to newly trained robust teachers:

```bash
TEACHER_PROGRESS_CKPT=... \
TEACHER_RULE_CKPT=... \
TEACHER_SAFETY_CKPT=... \
TEACHER_GENERAL_CKPT=... \
bash scripts/training/run_recogdrive_train_goalbridge_opd_8gpu.sh
```

If continuing GoalBridge training from a GoalBridge Student checkpoint, `ANCHOR_CKPT` remains fixed to `BASE_IL_CKPT` by default rather than silently following `STUDENT_CKPT`.

## 2. Train robust privileged teachers in parallel

The robust teacher uses one mutually-exclusive draw per sample:

- 70% clean GT goal
- 20% noisy goal
- 10% masked goal

Default noise: `1.0 m` for xy and `0.10 rad` for heading (heading noise is inert while `goal_use_heading=false`).

Run each expert independently:

```bash
bash scripts/training/run_recogdrive_train_robust_goal_teacher_progress.sh
bash scripts/training/run_recogdrive_train_robust_goal_teacher_rule.sh
bash scripts/training/run_recogdrive_train_robust_goal_teacher_safety.sh
bash scripts/training/run_recogdrive_train_robust_goal_teacher_general.sh
```

Each wrapper now defaults to **fine-tuning the corresponding existing strong privileged teacher**, with `LR=5e-5` and `MAX_EPOCHS=50`, rather than relearning the goal branch from the goal-free IL base. All values are overrideable.

Example:

```bash
GOAL_DROPOUT_P=0.10 \
GOAL_NOISE_P=0.20 \
GOAL_NOISE_STD_XY=1.0 \
LR=2e-5 \
MAX_EPOCHS=30 \
bash scripts/training/run_recogdrive_train_robust_goal_teacher_progress.sh
```

## Why v1 predicts a point, not clusters

v1 deliberately uses a continuous point head because it isolates the main hypothesis. If this recovers performance, the next clean extension is `goal prototype distribution + residual`, while keeping the same recoverability-aware shared-KL OPD framework.

## Files added

- `navsim/agents/recogdrive/recogdrive_predicted_goal_planner.py`
- `navsim/agents/recogdrive/recogdrive_goalbridge_distill_trainer.py`
- `navsim/agents/recogdrive/recogdrive_goalbridge_agent.py`
- `navsim/agents/recogdrive/recogdrive_robust_goal_teacher_agent.py`
- `navsim/planning/script/config/common/agent/recogdrive_agent_goalbridge_opd.yaml`
- `navsim/planning/script/config/common/agent/recogdrive_agent_robust_goal_teacher.yaml`
- `navsim/planning/script/run_training_recogdrive_robust_goal_teacher.py`
- GoalBridge and robust-teacher shell launchers under `scripts/training/`

Modified:

- `recogdrive_goal_planner.py`: adds mutually-exclusive clean/noisy/masked goal corruption, implemented without per-batch GPU synchronisation.
- `agent_lightning_module_scene_router_goal.py`: logs GoalBridge diagnostics.

## Validation performed

- all changed/new Python files pass `py_compile` / `compileall`;
- all new shell scripts pass `bash -n`;
- the shared-DDIM-mean helper was numerically checked against the direct DDIM formula (max error `0.0` in the smoke test);
- a CPU fake-planner integration test ran the complete `compute_loss -> backward` path with four routed teachers + predicted goal + shared-KL + recoverability + anchor, and verified finite non-zero gradients for both Goal Head and Student planning core;
- the artifact container lacks the full project dependencies (`transformers`, `hydra`, `timm`, `nuplan`) and GPUs, so a true ReCogDrive first-batch CUDA forward still needs to be run in your training environment before a long job.
