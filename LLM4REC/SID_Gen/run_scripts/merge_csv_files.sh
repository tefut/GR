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


echo "Running: $CMD"

# 执行命令
eval $CMD
# 检测上一步执行结果
if [ $? -ne 0 ]; then
    echo "步骤执行失败，正在退出..."
    exit 1
fi
echo "Done!"
