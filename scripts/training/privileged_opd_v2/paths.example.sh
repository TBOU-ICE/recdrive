#!/usr/bin/env bash
# Copy to paths.local.sh and fill these for your cluster. Scripts never edit old code.

export CONDA_BIN=/opt/conda/envs/recdrive/bin
export VLM_PATH=/PATH/TO/vlm_simscale_lora_merged

# Stage-1 common IL fullmix checkpoint (NAV full + SimScale).
export STAGE1_IL_CKPT=/PATH/TO/training_dit_il_fullmix_simscale.ckpt
# Recommended main comparison: initialize OPD student from the SAME post-IL
# common checkpoint used before the four scene-specific RL experts. This keeps
# common ancestry and reproduces the old successful IL->(RL teacher)->OPD setup.
export STUDENT_CKPT=${STAGE1_IL_CKPT}

# Old goal-free scene experts AFTER original IL->RL. These are stage-3 initializers.
export RL_PROGRESS_CKPT=/PATH/TO/progress_goal_free_rl.ckpt
export RL_RULE_CKPT=/PATH/TO/rule_goal_free_rl.ckpt
export RL_SAFETY_CKPT=/PATH/TO/safety_goal_free_rl.ckpt
export RL_GENERAL_CKPT=/PATH/TO/general_goal_free_rl.ckpt

# Fill after stage-3 goal-adapter training.
export PRIV_PROGRESS_CKPT=/PATH/TO/progress_goal_adapter.ckpt
export PRIV_RULE_CKPT=/PATH/TO/rule_goal_adapter.ckpt
export PRIV_SAFETY_CKPT=/PATH/TO/safety_goal_adapter.ckpt
export PRIV_GENERAL_CKPT=/PATH/TO/general_goal_adapter.ckpt

# Cached new-VLM hidden states + manifests.
export NAV_CACHE=/PATH/TO/recogdrive_agent_cache_dir_train
export NAV_MANIFEST=/PATH/TO/nav_train_newvlm.json
export NAV_BUCKET_ROOT=/PATH/TO/navtrain_scene/output/navtrain
export TOKEN_TO_BUCKET_NAV=${NAV_BUCKET_ROOT}/exclusive_token_to_bucket.json

export SIM_CACHE_R0=/PATH/TO/recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-0
export SIM_CACHE_R1=/PATH/TO/recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-1
export SIM_MANIFEST_R0=/PATH/TO/sim_round0_newvlm.json
export SIM_MANIFEST_R1=/PATH/TO/sim_round1_newvlm.json
export SIM_BUCKET_ROOT=/PATH/TO/simscale
# Expected files below each quality bucket directory:
#   exclusive_<bucket>_tokens.json and exclusive_token_to_bucket.json
export SIM_BUCKET_R0_ROOT=${SIM_BUCKET_ROOT}/scene_buckets_synthetic_reaction_pdm_v1.0-0_quality
export SIM_BUCKET_R1_ROOT=${SIM_BUCKET_ROOT}/scene_buckets_synthetic_reaction_pdm_v1.0-1_quality

# C/G only.
export GOAL_VOCAB=/PATH/TO/goal_vocab_2048.npz

# NAVSIM environment.
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/PATH/TO/maps/nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/PATH/TO/navsim/download
export NAVSIM_EXP_ROOT=/PATH/TO/exp
