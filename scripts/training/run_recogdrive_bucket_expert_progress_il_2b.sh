export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="/workspace/code"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is empty}"
MASTER_PORT="${MASTER_PORT:-23456}"
GPUS="${GPUS:-8}"

BASE_CKPT="${BASE_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt}"
BUCKET_JSON="/workspace/volumes/ad-e2e-al-sh01/nby/data/navtrain_scene/output/navtrain/exclusive_progress_curbside_stopgo_tokens.json"
CACHE_PATH="${CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train}"

/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
  --nnodes=${NNODES} \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_bucket_il.py \
  agent=recogdrive_agent \
  "agent.checkpoint_path=\"${BASE_CKPT}\"" \
  agent.lr=5e-5 \
  agent.grpo=False \
  agent.vlm_path='/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B' \
  agent.cam_type='single' \
  agent.cache_hidden_state=True \
  agent.vlm_type='internvl' \
  agent.dit_type='small' \
  agent.vlm_size='small' \
  agent.sampling_method='ddim' \
  trainer.params.max_epochs=8 \
  trainer.params.num_nodes=${NNODES} \
  trainer.params.devices=${GPUS} \
  experiment_name=training_recogdrive_bucket_progress_il \
  cache_path="${CACHE_PATH}" \
  bucket.name=progress_curbside_stopgo \
  bucket.tokens_json="${BUCKET_JSON}" \
  bucket.full_ratio=0.6 \
  bucket.bucket_ratio=0.4 \
  hydra/job_logging=stdout \
  hydra.output_subdir=null
