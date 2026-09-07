# ReCogDrive DiT-OPD changes

This patch adds a pure continuous DiT-OPD path for the setting:

- fixed/cached VLM hidden state shared by teacher and student
- teacher DiT: RL checkpoint
- student DiT: IL checkpoint
- training data: NAVSIM cached dataset
- loss: diffusion reverse-transition KL on the student's on-policy denoising chain

## Main files

- `navsim/agents/recogdrive/recogdrive_dit_opd_trainer.py`
  - New trainer implementing closed-form DDPM/DDIM Gaussian transition KL.
  - Student samples its own denoising chain.
  - Teacher is evaluated on the same student `z_t` states.
  - No VLM token OPD, no top-k, no PDM reward weighting, no feature/traj matching loss.

- `navsim/agents/recogdrive/recogdrive_agent.py`
  - Adds `dit_opd` mode.
  - Loads frozen teacher DiT from `teacher_dit_checkpoint`.
  - Loads trainable student DiT from `checkpoint_path` via existing `initialize()`.
  - Uses the same cached `last_hidden_state` for both teacher and student.
  - Adds checkpoint-file-or-directory resolution.

- `navsim/planning/training/agent_lightning_module.py`
  - Logs DiT-OPD metrics: `transition_kl`, `step_kl_mean`, `step_kl_max`, `pred_traj_l1_to_teacher`, `denoising_steps`.

- `navsim/planning/script/config/common/agent/recogdrive_agent_dit_opd.yaml`
  - New Hydra agent config for the requested teacher/student paths.

- `scripts/train_recogdrive_dit_opd.sh`
  - Example launch script. Override NAVSIM cache/output paths as needed.

## Loss

For each student denoising step, the loss is:

```text
KL(p_theta(z_{t-1} | z_t, h) || p_phi(z_{t-1} | z_t, h))
```

For Gaussian transitions, the implementation uses the full closed-form reverse KL. When variance is shared this reduces to:

```text
0.5 * ||mu_theta(z_t,t,h) - mu_phi(z_t,t,h)||^2 / sigma_t^2
```

where `h` is the cached VLM hidden state and `z_t` is sampled by the student.

## Example launch

```bash
bash scripts/train_recogdrive_dit_opd.sh
```

or directly:

```bash
python -m navsim.planning.script.run_training_recogdrive_rl \
  agent=recogdrive_agent_dit_opd \
  output_dir=/workspace/outputs/recogdrive_dit_opd \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  cache_path=/workspace/volumes/ad-e2e-bd-su01/nby/recdrive/training_cache_recogdrive
```


## Implementation notes checked

- DiT-OPD installs stochastic denoising defaults locally instead of relying on GRPO initialization.
- The transition KL uses the full Gaussian formula, so it remains valid if teacher/student DDIM variance differs.
- The default config does not hard-clamp `max_kl_per_step`; use external gradient clipping or set this manually only for debugging.

## v3 audit fixes

- Student checkpoint loading is now robust to both full-agent checkpoints (`agent.action_head.*`) and standalone DiT planner checkpoints (`model.*`, `feature_encoder.*`, etc.). Standalone planner keys are automatically mapped under `action_head.*`.
- The loader now prints how many action-head tensors were loaded and raises an error if zero planner tensors match, avoiding silent random-student training.
- The example DiT-OPD launch script now uses `torchrun`, sets the same distributed environment style as existing ReCogDrive scripts, validates the student/teacher/cache paths before launch, and exposes `STUDENT_CKPT`, `TEACHER_CKPT`, `CACHE_PATH`, `OUTPUT_DIR`, `GPUS`, and `NAVSIM_DEVKIT_ROOT` as environment overrides.
