#!/bin/bash
# Embedding拼接预处理运行脚本

CONFIG_FILE="${1:-configs/merge_embeddings.yaml}"
BASE_DIR="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "Project Root: $PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:$(dirname "$PROJECT_ROOT")

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
echo "Using config file: $PROJECT_ROOT/$CONFIG_FILE"

cd "$PROJECT_ROOT"

python preprocess/merge_embeddings.py --config "$CONFIG_FILE"

if [ $? -ne 0 ]; then
    echo "步骤执行失败，正在退出..."
    exit 1
fi
