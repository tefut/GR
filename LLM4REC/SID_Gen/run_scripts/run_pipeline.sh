#!/bin/bash
# RQVAE 流水线脚本
# 执行完整流程: Embedding生成 -> 训练 -> 评价

current_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$current_dir/.." && pwd)"
export PYTHONPATH="$project_root:$PYTHONPATH"

# 环境设置
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

export ASCEND_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# 默认使用 game.yaml 配置
CONFIG_FILE="${1:-configs/game.yaml}"

echo "=========================================="
echo "RQVAE Pipeline Runner"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Project Root: $project_root"
echo "=========================================="

# 使用 Python 运行流水线
python -c "
import sys
sys.path.insert(0, '$project_root')
from pipeline.runner import PipelineRunner

runner = PipelineRunner(
    config_path='$CONFIG_FILE',
    log_level='INFO',
)
results = runner.run_all()
print('Pipeline Results:', results)
sys.exit(1 if any(c != 0 for c in results.values()) else 0)
"
