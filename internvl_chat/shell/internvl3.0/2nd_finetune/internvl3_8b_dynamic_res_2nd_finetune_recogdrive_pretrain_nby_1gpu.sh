set -x

# Single-GPU bootstrap config: prioritize successful startup.
GPUS=${GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
if [ "${GRADIENT_ACC}" -lt 1 ]; then
  GRADIENT_ACC=1
fi
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}

NNODES=${NNODES:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-34229}

# Use the first visible GPU by default.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"

if command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1 && command -v x86_64-conda-linux-gnu-g++ >/dev/null 2>&1; then
  export CC="${CC:-x86_64-conda-linux-gnu-gcc}"
  export CXX="${CXX:-x86_64-conda-linux-gnu-g++}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-x86_64-conda-linux-gnu-g++}"
fi

OUTPUT_DIR=${OUTPUT_DIR:-'/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/outputs/ReCogDrive_pretrain/all_data_1gpu_smoke'}
mkdir -p "$OUTPUT_DIR"

/opt/conda/envs/recdrive/bin/torchrun \
  --standalone \
  --nnodes=${NNODES} \
  --nproc_per_node=${GPUS} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/InternVL3-2B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data_info/recogdrive_pretrain.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 4 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
  --bf16 True \
  --fp16 False \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 3 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
