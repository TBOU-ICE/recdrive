#!/bin/bash

set -x

# 强制使用指定 conda 环境（不依赖 activate，兼容所有机器）
# 部分调度器/误配会把 URL（如 http://IP:PORT）写进 PATH；按 ":" 切分后会变成 //IP、tcp、80 等假路径，
# Accelerate 等库校验 PATH 时会报错。这里先去掉这些片段再 prepend conda。
internvl_sanitize_path() {
  local _p="$1" _d _out="" 
  _p="${_p}:"
  while [ -n "$_p" ]; do
    _d="${_p%%:*}"
    _p="${_p#*:}"
    [ -z "$_d" ] && continue
    case "$_d" in
      *"://"*|http://*|https://*|tcp://*) continue ;;
      //*) continue ;;
      tcp) continue ;;
    esac
    [[ "$_d" =~ ^[0-9]+$ ]] && continue
    _out="${_out+${_out}:}${_d}"
  done
  printf '%s' "$_out"
}
_REC_DRIVE_BIN="/mnt/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin"
export PATH="${_REC_DRIVE_BIN}:$(internvl_sanitize_path "$PATH")"

# 验证（可选，调试用）
echo "Using python: $(which python)"
echo "Python version: $(python --version)"
PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
if [ "${GRADIENT_ACC}" -lt 1 ]; then
  GRADIENT_ACC=1
fi
# Start from 0 for stability, then tune up (1/2/4) if resources allow.
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}

NNODES="${WORLD_SIZE:?WORLD_SIZE is empty}"
RANK="${RANK:?RANK is empty}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is empty}"
MASTER_PORT="${MASTER_PORT:?MASTER_PORT is empty}"
GPUS="${GPUS:-8}"  


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"

if command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1 && command -v x86_64-conda-linux-gnu-g++ >/dev/null 2>&1; then
  export CC="${CC:-x86_64-conda-linux-gnu-gcc}"
  export CXX="${CXX:-x86_64-conda-linux-gnu-g++}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-x86_64-conda-linux-gnu-g++}"
fi

OUTPUT_DIR='/mnt/volumes/ad-e2e-bd-su01/nby/recdrive/outputs/ReCogDrive_pretrain/all_data'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

if ! python - <<'PY'
from deepspeed.ops.op_builder import FusedAdamBuilder
FusedAdamBuilder().load(verbose=False)
print('[env-check] fused_adam extension is ready')
PY
then
  echo "[env-check] fused_adam extension build failed. Please check gcc/g++ and CUDA compatibility." >&2
  exit 1
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1

  # --nnodes=8 \
  # --node_rank=$MLP_ROLE_INDEX \
  # --master_addr=$MLP_WORKER_0_HOST \
  # --master_port=$MLP_WORKER_0_PORT \
/mnt/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun \
  --nnodes=${NNODES} \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  --nproc_per_node=${GPUS} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "/mnt/models/recdrive/v1.0.0/InternVL3-2B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data_info/recogdrive_pretrain.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 16 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 10 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 12288 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
