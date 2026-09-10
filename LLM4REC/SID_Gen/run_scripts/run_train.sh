#!/bin/bash
# Copyright 2026 作者：灵犀
# 语义ID训练任务运行脚本

# 配置CANN相关环境变量
echo -e "\n====== 配置CANN相关环境变量 ======"
CANN_INSTALL_PATH_CONF='/etc/Ascend/ascend_cann_install.info'
sudo cat /etc/Ascend/ascend_cann_install.info
DEFAULT_CANN_INSTALL_PATH="/usr/local/Ascend/"
CANN_INSTALL_PATH="/usr/local/Ascend/"

if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ];then
  cat ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
  source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
  cat ${CANN_INSTALL_PATH}/nnae/set_env.sh
  source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi
echo -e "\n====== CANN相关环境变量配置完成 ======"

pip install scikit-learn 2>/dev/null wandb
pip install transformers==5.10.2 peft==0.19.1 accelerate==1.13.0 torch==2.7.1 torch-npu==2.7.1 torchvision
pip install accelerate faiss-cpu k_means_constrained polars "numpy<2.0.0"
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
export PYTHONPATH=$PYTHONPATH:$(dirname "$PROJECT_ROOT")

# 默认参数
CONFIG_FILE="${PROJECT_ROOT}/configs/train_sid.yaml"

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
echo "SID Training Task"
echo "=========================================="
echo "Config:   $CONFIG_FILE"
echo "=========================================="

cd "$PROJECT_ROOT"

# 运行任务
accelerate launch \
    --mixed_precision fp16 \
    train_sid/train_sid.py \
    --config "$CONFIG_FILE"
# 检测上一步执行结果
if [ $? -ne 0 ]; then
    echo "步骤执行失败，正在退出..."
    exit 1
fi
echo "Done!"
