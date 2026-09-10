#!/bin/bash
# Copyright 2026 作者：灵犀
# Merge App SID 任务运行脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Project Root: $PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:$(dirname "$PROJECT_ROOT")
# 默认参数
CONFIG_FILE="${PROJECT_ROOT}/configs/merge_app_sid.yaml"
pip install scikit-learn
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
echo "Merge App SID Task"
echo "=========================================="
echo "Config:   $CONFIG_FILE"
echo "=========================================="

cd "$PROJECT_ROOT"

python scripts/merge_app_sid_to_parquet.py \
    --config "$CONFIG_FILE"
# 检测上一步执行结果
if [ $? -ne 0 ]; then
    echo "步骤执行失败，正在退出..."
    exit 1
fi
echo "Done!"
