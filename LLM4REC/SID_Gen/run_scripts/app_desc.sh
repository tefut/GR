#!/bin/bash
# Copyright 2026 作者：灵犀
# 应用描述生成任务运行脚本
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
pip install transformers accelerate
pip install sentencepiece
pip install pandas
pip install mistral-common>=1.6.0
export HCCL_CONNECT_TIMEOUT=600
export HCCL_WHITELIST_DISABLE=1
NGPUS_PER_NODE="${MA_NUM_GPUS: -1}"
echo "[INFO] Using ${NGPUS_PER_NODE} NPUs"
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Project Root: $PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:$(dirname "$PROJECT_ROOT")

# 默认参数
# /opt/huawei/schedule-train/algorithm/train.config
CONFIG_FILE="${PROJECT_ROOT}/configs/llm/app_desc.yaml"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        # 🔹 兼容等号格式：--config=/path/to/config.yaml
        --config=*)
            VAL="${1#*=}"
            if [[ -n "$VAL" ]]; then
                CONFIG_FILE="$VAL"
            fi
            shift
            ;;
        # 🔹 兼容空格格式：--config /path/to/config.yaml
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
# ==========================================

# 自动检测设备
if python3 -c "import torch; import torch_npu; assert torch.npu.is_available()" 2>/dev/null; then
    DEVICE="npu"
    MIXED_PRECISION="fp16"
elif python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    MIXED_PRECISION="fp16"
else
    DEVICE="cpu"
    MIXED_PRECISION="no"
fi

echo "=========================================="
echo "App Description Generation Task"
echo "=========================================="
echo "Config:   $CONFIG_FILE"
echo "Device:   $DEVICE"
echo "=========================================="

cd "$PROJECT_ROOT"

# 运行任务
accelerate launch \
    --mixed_precision $MIXED_PRECISION \
    llm_generation/llm_generate.py \
    --config "$CONFIG_FILE"
# 检测上一步执行结果
if [ $? -ne 0 ]; then
    echo "步骤执行失败，正在退出..."
    exit 1
fi
echo "Done!"
