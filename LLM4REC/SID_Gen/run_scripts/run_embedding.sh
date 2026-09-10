#!/bin/bash
# Copyright 2026 作者：灵犀
# Embedding生成任务运行脚本

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

pip install pandas qwen-vl-utils mistral-common 2>/dev/null
pip install transformers==5.10.2 accelerate==1.13.0 torch==2.7.1 torch-npu==2.7.1 torchvision
export HCCL_CONNECT_TIMEOUT=600
export HCCL_WHITELIST_DISABLE=1
export ASCEND_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

NGPUS_PER_NODE="${MA_NUM_GPUS: -1}"
echo "[INFO] Using ${NGPUS_PER_NODE} NPUs"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Project Root: $PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:"$PROJECT_ROOT"

# 默认参数
CONFIG_FILE="${PROJECT_ROOT}/configs/embed_gen.yaml"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config=*)
            VAL="${1#*=}"
            if [[ -n "$VAL" ]]; then
                CONFIG_FILE="$VAL"
            fi
            shift
            ;;
        --config)
            if [[ $# -ge 2 && -n "$2" ]]; then
                CONFIG_FILE="$2"
                shift 2
            else
                shift
            fi
            ;;
        *)
            echo "Warning: 未知参数 '$1'，已忽略。使用默认配置。"
            shift
            ;;
    esac
done

echo "=========================================="
echo "Embedding Generation Task"
echo "=========================================="
echo "Config:   $CONFIG_FILE"
echo "=========================================="

cd "$PROJECT_ROOT"

# 运行任务
accelerate launch \
    --mixed_precision fp16 \
    --dynamo_backend inductor \
    embedding_generation/emb_generate_txt.py \
    --config "$CONFIG_FILE"
# 检测上一步执行结果
if [ $? -ne 0 ]; then
    echo "步骤执行失败，正在退出..."
    exit 1
fi
echo "Done!"
