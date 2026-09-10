#!/bin/bash
# Copyright 2026 作者：灵犀
# SID Evaluation Task Run Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Project Root: $PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
pip install scikit-learn
# Default config
CONFIG_FILE="${PROJECT_ROOT}/configs/eval_sid.yaml"

# Parse parameters
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
            echo "Warning: Unknown parameter '$1', ignored. Using default config."
            shift
            ;;
    esac
done

echo "=========================================="
echo "SID Evaluation Task"
echo "=========================================="
echo "Config:   $CONFIG_FILE"
echo "=========================================="

cd "$PROJECT_ROOT"

python eval_sid/sid_eval_unified.py \
    --config "$CONFIG_FILE"
# 检测上一步执行结果
if [ $? -ne 0 ]; then
    echo "步骤执行失败，正在退出..."
    exit 1
fi
echo "Done!"
