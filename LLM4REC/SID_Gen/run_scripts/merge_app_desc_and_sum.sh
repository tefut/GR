#!/bin/bash
# Copyright 2026-2026
# 游戏数据合并与评论预处理运行脚本

CONFIG_FILE="${1:-configs/merge_app_and_sum.yaml}"
BASE_DIR="${2:-}"

# 检查配置文件是否存在
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "Project Root: $PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:$(dirname "$PROJECT_ROOT")
pip install json5 scikit-learn

# 2. 解析命令行参数
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
echo "Using config file: $PROJECT_ROOT/$CONFIG_FILE"

# 运行Python脚本
cd "$PROJECT_ROOT"

python preprocess/merge_app_desc_and_sum.py --config "$CONFIG_FILE"

# 检测上一步执行结果
if [ $? -ne 0 ]; then
    echo "步骤执行失败，正在退出..."
    exit 1
fi
