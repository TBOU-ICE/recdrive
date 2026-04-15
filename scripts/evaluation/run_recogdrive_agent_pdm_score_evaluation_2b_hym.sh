#!/bin/bash
# Invoked as `sh this.sh` uses dash: no arrays, no <<<. Re-exec under bash.
if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi
set -x

# ========== 公共环境变量 ==========
# Same interpreter as run_recogdrive_agent_pdm_score_evaluation_2b_1gpu_debug.sh (recdrive conda env).
RECDRIVE_CONDA_BIN="${RECDRIVE_CONDA_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin}"
TORCHRUN="${RECDRIVE_CONDA_BIN}/torchrun"
export PATH="${RECDRIVE_CONDA_BIN}:${PATH}"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTHONPATH="/workspace/recdrive:${PYTHONPATH:-}"


TRAIN_TEST_SPLIT=navtest
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
WORKER_THREADS_PER_NODE=${WORKER_THREADS_PER_NODE:-8}

export CUDA_LAUNCH_BLOCKING=1

# ========== 评估任务列表 ==========
EVAL_TASKS=(
    "recdrive_rl_nby|/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt|/workspace/output/recogdrive_rl"
)

# ========== 循环执行 ==========
for task in "${EVAL_TASKS[@]}"; do
    IFS='|' read -r exp_name ckpt_path output_dir <<< "$task"
    
    echo "=========================================="
    echo "开始评估: $exp_name"
    echo "Checkpoint: $ckpt_path"
    echo "Output: $output_dir"
    echo "=========================================="
    
    # 生成随机端口
    MASTER_PORT=$((29500 + RANDOM % 1000))
    
    # 关键修复1: checkpoint_path 值包含 '='，必须用引号包裹整个参数
    # 关键修复2: output_dir 同理需要引号（路径中可能有特殊字符）
    # 关键修复3: 不使用 \ 续行符，避免解析错误
    "${TORCHRUN}" \
        --nproc_per_node=1 \
        --master_port=$MASTER_PORT \
        /workspace/recdrive/navsim/planning/script/run_pdm_score_recogdrive.py \
        train_test_split=$TRAIN_TEST_SPLIT \
        agent=recogdrive_agent \
        "agent.checkpoint_path='$ckpt_path'" \
        "agent.vlm_path='/workspace/volumes/ad-e2e-al-sh01/hym/recdrive/ReCogDrive-VLM-2B'" \
        agent.cam_type=single \
        agent.grpo=False \
        agent.cache_hidden_state=False \
        agent.vlm_type=internvl \
        agent.dit_type=small \
        agent.vlm_size=small \
        agent.sampling_method=ddim \
        experiment_name=recogdrive_eval_${exp_name} \
        worker.threads_per_node=$WORKER_THREADS_PER_NODE \
        worker.log_to_driver=false \
        "output_dir='$output_dir'"
    
    if [ $? -eq 0 ]; then
        echo "✅ $exp_name 评估完成"
    else
        echo "❌ $exp_name 评估失败，错误码: $?"
    fi
    
    sleep 5
done

echo "所有评估任务完成！"