#!/bin/bash

train_workdir=$(cd $(dirname $0); pwd)
echo ${train_workdir}


# 设置默认值
MASTER_PORT="${MASTER_PORT:-29400}"
DEFAULT_MASTER="127.0.0.1"

# 根据不同环境获取变量
if [[ -n "${SLURM_PROCID}" ]]; then
    echo "SLURM_PROCID"
    # SLURM环境
    MASTER_ADDR="${SLURM_LAUNCH_NODE_IPADDR:-$DEFAULT_MASTER}"
    NODE_RANK="${SLURM_PROCID}"
    LOCAL_RANK="${SLURM_LOCALID}"
    WORLD_SIZE="${SLURM_NTASKS}"
    LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE}"
elif [[ -n "${VC_WORKER_HOSTS}" ]]; then
    echo "VC_WORKER_HOSTS"
    # 自定义集群环境
    IFS=',' read -ra NODE_ARRAY <<< "$VC_WORKER_HOSTS"
    MASTER_ADDR="${NODE_ARRAY[0]:-$DEFAULT_MASTER}"
    CURRENT_HOST="$(hostname)"

    NODE_RANK="0"  # 默认为主节点
    for i in "${!NODE_ARRAY[@]}"; do
        if [[ "${NODE_ARRAY[$i]}" == "$CURRENT_HOST" ]]; then
            NODE_RANK="$i"
            break
        fi
    done

    WORLD_SIZE="${#NODE_ARRAY[@]}"
    LOCAL_RANK="0"  # 需根据实际情况设置
    LOCAL_WORLD_SIZE="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
else
    echo "single"
    # 单机环境
    MASTER_ADDR="$DEFAULT_MASTER"
    NODE_RANK="0"
    LOCAL_RANK="0"
    WORLD_SIZE="1"
    LOCAL_WORLD_SIZE="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
fi

# 构建主节点端点
MASTER_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"

# 输出信息
echo "======== 分布式训练环境配置 ========"
echo "主节点地址: ${MASTER_ADDR}"
echo "主节点端口: ${MASTER_PORT}"
echo "主节点端点: ${MASTER_ENDPOINT}"
echo "当前节点排名: ${NODE_RANK}"
echo "当前本地排名: ${LOCAL_RANK}"
echo "总节点数/机器数: ${WORLD_SIZE}"
echo "每个节点的进程数: ${LOCAL_WORLD_SIZE}"
echo "===================================="


torchrun \
    --nnodes="${WORLD_SIZE}" \
    --nproc_per_node="${LOCAL_WORLD_SIZE}" \
    --rdzv_id="my_training_job" \
    --rdzv_backend="c10d" \
    --rdzv_endpoint="${MASTER_ENDPOINT}" \
    ${train_workdir}/recall_main.py --device "gpu"
