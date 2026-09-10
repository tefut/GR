#!/bin/bash
# Copyright 2026 作者：灵犀
# CSV表头添加任务运行脚本

pip install pandas numpy pyyaml 2>/dev/null

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Project Root: $PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:"$PROJECT_ROOT"

CONFIG_FILE="${PROJECT_ROOT}/configs/add_csv_header.yaml"

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
            echo "Warning: 未知参数 '$1'，已忽略。"
            shift
            ;;
    esac
done

echo "=========================================="
echo "Add CSV Header Task"
echo "=========================================="
echo "Config:   $CONFIG_FILE"
echo "=========================================="

cd "$PROJECT_ROOT"

CMD="python preprocess/add_csv_header.py --config \"$CONFIG_FILE\""

echo "Running: $CMD"

eval $CMD
if [ $? -ne 0 ]; then
    echo "步骤执行失败，正在退出..."
    exit 1
fi
echo "Done!"
