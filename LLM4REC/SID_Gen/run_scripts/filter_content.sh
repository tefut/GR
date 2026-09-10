#!/bin/bash
# Copyright 2026 作者：灵犀
# 内容过滤任务运行脚本

pip install pandas numpy pyyaml 2>/dev/null

# 获取脚本在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Project Root: $PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:"$PROJECT_ROOT"

# 默认参数
CONFIG_FILE="${PROJECT_ROOT}/configs/filter_content.yaml"

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
        --input_file=*)
            INPUT_FILE="${1#*=}"
            shift
            ;;
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output_file=*)
            OUTPUT_FILE="${1#*=}"
            shift
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --input_column=*)
            INPUT_COLUMN="${1#*=}"
            shift
            ;;
        --input_column)
            INPUT_COLUMN="$2"
            shift 2
            ;;
        --filter_function=*)
            FILTER_FUNCTION="${1#*=}"
            shift
            ;;
        --filter_function)
            FILTER_FUNCTION="$2"
            shift 2
            ;;
        *)
            echo "Warning: 未知参数 '$1'，已忽略。"
            shift
            ;;
    esac
done

echo "=========================================="
echo "Content Filter Task"
echo "=========================================="
echo "Config:   $CONFIG_FILE"
echo "=========================================="

cd "$PROJECT_ROOT"

# 构建命令
CMD="python scripts/filter_content.py --config \"$CONFIG_FILE\""

# 添加可选参数
if [[ -n "$INPUT_FILE" ]]; then
    CMD="$CMD --input_file \"$INPUT_FILE\""
fi
if [[ -n "$OUTPUT_FILE" ]]; then
    CMD="$CMD --output_file \"$OUTPUT_FILE\""
fi
if [[ -n "$INPUT_COLUMN" ]]; then
    CMD="$CMD --input_column \"$INPUT_COLUMN\""
fi
if [[ -n "$FILTER_FUNCTION" ]]; then
    CMD="$CMD --filter_function \"$FILTER_FUNCTION\""
fi

echo "Running: $CMD"

# 执行命令
eval $CMD
# 检测上一步执行结果
if [ $? -ne 0 ]; then
    echo "步骤执行失败，正在退出..."
    exit 1
fi
echo "Done!"
