#!/usr/bin/env bash
# Phase B: student SFT on the Phase A teacher-rollout cache.
#
# The student is the same PredictedGoalDiffusionPlanner the GoalBridge OPD run
# trains, so the checkpoint produced here goes straight into STUDENT_CKPT of
# run_recogdrive_train_goalbridge_opd_8gpu.sh.
#
# Objective: epsilon-MSE onto a teacher rollout, conditioned on that rollout's
# OWN endpoint (teacher forcing, self-consistent pair), plus SmoothL1 on the
# goal head. The two touch disjoint parameters, so goal_loss_weight is not a
# sensitive knob.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/opt/conda/envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/workspace/output/tensorboard}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1
export SCENE_ROUTER_BAD_CACHE_LIST="${SCENE_ROUTER_BAD_CACHE_LIST:-${REPO_ROOT}/data/epdms/bad_cache_shards_newvlm.txt}"

NNODES="${NNODES:-1}"; RANK="${RANK:-0}"; GPUS="${GPUS:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"; MASTER_PORT="${MASTER_PORT:-23532}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

TEACHER_ROLLOUT_DIR="${TEACHER_ROLLOUT_DIR:-${NAVSIM_EXP_ROOT}/teacher_rollout_cache_adaln}"

# Same goal-free IL base the OPD run starts from; SFT is what closes the gap.
BASE_IL_CKPT="${BASE_IL_CKPT:-/workspace/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
STUDENT_CKPT="${STUDENT_CKPT:-${BASE_IL_CKPT}}"
RESUME_CKPT="${RESUME_CKPT:-}"

VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
# Same direct indexes the IL bucket experts use: absolute token-dir paths with
# navtrain + simscale already merged. No manifest, no quality whitelist, no links.
DIRECT_INDEX_ROOT="${DIRECT_INDEX_ROOT:-/mnt/datasets/simscale/20260709/data/simscale/il_training_newvlm_direct}"
INDEX_NAME="${INDEX_NAME:-train_index.json}"
VAL_INDEX_NAME="${VAL_INDEX_NAME:-navtrain_full_val_index.json}"

GOAL_LOSS_WEIGHT="${GOAL_LOSS_WEIGHT:-1.0}"
GOAL_TARGET="${GOAL_TARGET:-gt}"
GOAL_DROPOUT_P="${GOAL_DROPOUT_P:-0.10}"
GOAL_NOISE_P="${GOAL_NOISE_P:-0.30}"
GOAL_NOISE_STD_XY="${GOAL_NOISE_STD_XY:-2.0}"
MIN_ROLLOUT_COVERAGE="${MIN_ROLLOUT_COVERAGE:-0.90}"
GOAL_SENSITIVITY_INTERVAL="${GOAL_SENSITIVITY_INTERVAL:-0}"

# SFT is a warmup on static offline labels, i.e. pure off-policy. At 8x16 one
# epoch is ~1560 steps over the ~200k-scene union (the four bucket teachers sum
# to 1562 steps/epoch at this config, and the IL base ckpt's 4683 steps / 3
# epochs agrees). This is a CAP, not a target: checkpoints are written every
# epoch, so run it out and pick by the val/student_fde_teacher_m curve. Training
# to convergence memorises a fixed teacher output and costs OPD plasticity.
LR="${LR:-1e-4}"; MAX_EPOCHS="${MAX_EPOCHS:-10}"; BATCH_SIZE="${BATCH_SIZE:-16}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_teacher_sft_goalbridge_student_v1}"
LOG_FILE="${LOG_FILE:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/run.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
HYDRA_RESUME=(); [[ -n "${RESUME_CKPT}" ]] && HYDRA_RESUME+=("+ckpt_path='${RESUME_CKPT}'")

if ! ls "${TEACHER_ROLLOUT_DIR}"/teacher_rollout_shard_*.pt >/dev/null 2>&1; then
  echo "[TeacherSFT] ERROR: no rollout shards under ${TEACHER_ROLLOUT_DIR}." >&2
  echo "             Run scripts/generation/run_teacher_rollout_cache_8gpu.sh first." >&2
  exit 1
fi

echo "[TeacherSFT] student=${STUDENT_CKPT} rollouts=${TEACHER_ROLLOUT_DIR} goal_target=${GOAL_TARGET}"
echo "[TeacherSFT] direct_index_root=${DIRECT_INDEX_ROOT}"
echo "[TeacherSFT] corruption dropout=${GOAL_DROPOUT_P} noise=${GOAL_NOISE_P}@${GOAL_NOISE_STD_XY}m epochs=${MAX_EPOCHS}"

torchrun --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_teacher_sft.py" \
  agent=recogdrive_agent_teacher_sft \
  "agent.checkpoint_path='${STUDENT_CKPT}'" "agent.vlm_path='${VLM_PATH}'" \
  "agent.teacher_rollout_dir='${TEACHER_ROLLOUT_DIR}'" \
  agent.goal_loss_weight="${GOAL_LOSS_WEIGHT}" "agent.goal_target='${GOAL_TARGET}'" \
  agent.goal_dropout_p="${GOAL_DROPOUT_P}" agent.goal_noise_p="${GOAL_NOISE_P}" \
  agent.goal_noise_std_xy="${GOAL_NOISE_STD_XY}" \
  agent.min_rollout_coverage="${MIN_ROLLOUT_COVERAGE}" \
  agent.goal_sensitivity_interval="${GOAL_SENSITIVITY_INTERVAL}" \
  agent.lr="${LR}" \
  "+teacher_sft_direct_index_root='${DIRECT_INDEX_ROOT}'" \
  "+teacher_sft_index_name='${INDEX_NAME}'" "+teacher_sft_val_index_name='${VAL_INDEX_NAME}'" \
  trainer.params.max_epochs="${MAX_EPOCHS}" trainer.params.precision=bf16-mixed trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" trainer.params.strategy=ddp_find_unused_parameters_true \
  dataloader.params.batch_size="${BATCH_SIZE}" dataloader.params.num_workers=8 dataloader.params.prefetch_factor=4 \
  +dataloader.params.persistent_workers=true experiment_name="${EXPERIMENT_NAME}" train_test_split="${TRAIN_TEST_SPLIT}" \
  use_cache_without_dataset=True force_cache_computation=False \
  ${HYDRA_RESUME[@]+"${HYDRA_RESUME[@]}"} hydra/job_logging=stdout hydra.output_subdir=null 2>&1 | tee "${LOG_FILE}"

echo "[TeacherSFT] done. Acceptance checks:"
echo "  1. val/student_fde_teacher_m falls, then FLATTENS -- stop there, do not drive it to 0"
echo "  2. train/goal_encoder_wnorm left 0  -> the goal branch actually got gradient"
echo "  3. with GOAL_SENSITIVITY_INTERVAL>0, val/goal_sensitivity_m is clearly non-zero"
echo "     -> the goal changes the plan, not just the weights"
echo "Then point STUDENT_CKPT of run_recogdrive_train_goalbridge_opd_8gpu.sh at the best ckpt."
