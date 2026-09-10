#!/bin/bash
# Copyright 2026-2026
# 游戏数据合并描述信息运行脚本

CONFIG_FILE="${1:-configs/merge_desc.yaml}"
BASE_DIR="${2:-}"

# 检查配置文件是否存在
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $PROJECT_ROOT/$CONFIG_FILE"
    exit 1
fi

echo "Using config file: $PROJECT_ROOT/$CONFIG_FILE"

# 运行Python脚本
cd "$PROJECT_ROOT"

if [ -n "$BASE_DIR" ]; then
    python preprocess/merge_desc.py --config "$CONFIG_FILE" --base_dir "$BASE_DIR"
else
    python preprocess/merge_desc.py --config "$CONFIG_FILE"
fi
# 检测上一步执行结果
if [ $? -ne 0 ]; then
    echo "步骤执行失败，正在退出..."
    exit 1
fi
