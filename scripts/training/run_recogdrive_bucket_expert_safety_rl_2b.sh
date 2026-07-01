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

INIT_CKPT="${INIT_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_bucket_safety_il/2026.07.01.10.51.35/lightning_logs/version_0/checkpoints/epoch=5-step=3474.ckpt}"
REF_CKPT="${REF_CKPT:-${INIT_CKPT}}"
BUCKET_JSON="/workspace/volumes/ad-e2e-al-sh01/nby/data/navtrain_scene/output/navtrain/exclusive_safety_dynamics_interaction_tokens.json"
NAVTRAIN_OUTPUT_DIR="/workspace/volumes/ad-e2e-al-sh01/nby/data/navtrain_scene/output/navtrain"
CACHE_PATH="${CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/metric_cache_train}"

/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
  --nnodes=${NNODES} \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_bucket_rl.py \
  agent=recogdrive_agent \
  "agent.checkpoint_path=\"${INIT_CKPT}\"" \
  "agent.reference_policy_checkpoint=\"${REF_CKPT}\"" \
  agent.lr=1e-4 \
  agent.grpo=True \
  agent.vlm_path='/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B' \
  agent.cam_type='single' \
  agent.cache_hidden_state=True \
  agent.vlm_type='internvl' \
  agent.dit_type='small' \
  agent.vlm_size='small' \
  agent.sampling_method='ddim' \
  agent.metric_cache_path="${METRIC_CACHE_PATH}" \
  trainer.params.max_epochs=10 \
  trainer.params.num_nodes=${NNODES} \
  trainer.params.devices=${GPUS} \
  experiment_name=training_recogdrive_bucket_safety_rl \
  train_test_split=navtrain \
  cache_path="${CACHE_PATH}" \
  bucket.name=safety_dynamics_interaction \
  bucket.tokens_json="${BUCKET_JSON}" \
  bucket.token_to_log_json="${NAVTRAIN_OUTPUT_DIR}/navtrain_token_to_buckets.json" \
  bucket.navtrain_output_dir="${NAVTRAIN_OUTPUT_DIR}" \
  bucket.full_ratio=0.2 \
  bucket.bucket_ratio=0.8 \
  hydra/job_logging=stdout \
  hydra.output_subdir=null
