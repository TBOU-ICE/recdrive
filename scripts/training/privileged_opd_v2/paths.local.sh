#!/usr/bin/env bash

# Goal-free scene experts used to initialize Stage-3 privileged teachers.
export RL_GENERAL_CKPT=/workspace/models/recdrive/v1.0.0/training_teacher_general_or_no_tag_rl_newvlm/2026.07.24.09.07.23/lightning_logs/version_0/checkpoints/epoch=38-step=6825.ckpt
export RL_PROGRESS_CKPT=/workspace/models/recdrive/v1.0.0/training_teacher_progress_curbside_stopgo_rl_newvlm/2026.07.24.05.09.01/lightning_logs/version_0/checkpoints/epoch=38-step=6201.ckpt
export RL_RULE_CKPT=/workspace/models/recdrive/v1.0.0/training_teacher_rule_intersection_rl_newvlm/2026.07.24.04.59.12/lightning_logs/version_0/checkpoints/epoch=38-step=5499.ckpt
export RL_SAFETY_CKPT=/workspace/models/recdrive/v1.0.0/training_teacher_safety_dynamics_interaction_rl_newvlm/2026.07.24.09.41.09/lightning_logs/version_0/checkpoints/epoch=39-step=5040.ckpt
