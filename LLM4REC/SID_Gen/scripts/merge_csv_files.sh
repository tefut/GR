#!/bin/bash
# Copyright 2026 作者：灵犀
# CSV合并任务运行脚本

pip install pandas numpy pyyaml 2>/dev/null

# 获取脚本所目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Project Root: $PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:"$PROJECT_ROOT"

# 默认参数
CONFIG_FILE="${PROJECT_ROOT}/configs/merge_csv.yaml"

# 初始化参数变量
INPUT_FILES_STR=""
INPUT_COLUMNS_STR=""
OUTPUT_FILE=""
PRIMARY_KEY=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config=*)
            CONFIG_FILE="${1#*=}"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --input_files=*)
            INPUT_FILES_STR="${1#*=}"
            shift
            ;;
        --input_files)
            shift
            INPUT_FILES_STR=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                INPUT_FILES_STR="$INPUT_FILES_STR,$1"
                shift
            done
            ;;
        --input_columns=*)
            INPUT_COLUMNS_STR="${1#*=}"
            shift
            ;;
        --input_columns)
            shift
            INPUT_COLUMNS_STR=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                INPUT_COLUMNS_STR="$INPUT_COLUMNS_STR,$1"
                shift
            done
            ;;
        --output_file=*)
            OUTPUT_FILE="${1#*=}"
            shift
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --primary_key=*)
            PRIMARY_KEY="${1#*=}"
            shift
            ;;
        --primary_key)
            PRIMARY_KEY="$2"
            shift 2
            ;;
        --how=*)
            HOW="${1#*=}"
            shift
            ;;
        --how)
            HOW="$2"
            shift 2
            ;;
        *)
            echo "Warning: 未知参数 '$1'，已忽略。"
            shift
            ;;
    esac
done

echo "=========================================="
echo "Merge CSV Files Task"
echo "=========================================="
echo "Config:   $CONFIG_FILE"
echo "=========================================="

cd "$PROJECT_ROOT"

# 构建命令
CMD="python scripts/merge_csv_files.py --config \"$CONFIG_FILE\""

# 添加可选参数
if [[ -n "$INPUT_FILES_STR" ]]; then
    CMD="$CMD --input_files_str \"$INPUT_FILES_STR\""
fi
if [[ -n "$INPUT_COLUMNS_STR" ]]; then
    CMD="$CMD --input_columns_str \"$INPUT_COLUMNS_STR\""
fi
if [[ -n "$OUTPUT_FILE" ]]; then
    CMD="$CMD --output_file \"$OUTPUT_FILE\""
fi
if [[ -n "$PRIMARY_KEY" ]]; then
    CMD="$CMD --primary_key \"$PRIMARY_KEY\""
fi
if [[ -n "$HOW" ]]; then
    CMD="$CMD --how \"$HOW\""
fi

echo "Running: $CMD"

# 执行命令
eval $CMD

echo "Done!"
