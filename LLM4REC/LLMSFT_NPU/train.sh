#!/bin/bash

pip install transformers==4.31.0
pip install peft==0.4.0
pip install sentencepiece
pip install pandas

# 配置CANN相关环境变量
CANN_INSTALL_PATH_CONF='/etc/Ascend/ascend_cann_install.info'
DEFAULT_CANN_INSTALL_PATH="/usr/local/Ascend/"
CANN_INSTALL_PATH="/usr/local/Ascend/"

if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ];then
  cat ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
  source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
  cat ${CANN_INSTALL_PATH}/nnae/set_env.sh
  source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi

# 配置自定义环境变量
export HCCL_WHITELIST_DISABLE=1

# log
export ASCEND_SLOG_PRINT_TO_STDOUT=0   # 日志打屏, 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       # 日志级别常用 1 INFO级别; 3 ERROR级别
export ASCEND_GLOBAL_EVENT_ENABLE=0    # 默认不使能event日志信息
export ASCEND_LAUNCH_BLOCKING=1

# 系统默认环境变量，不建议修改
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NNODES="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
NGPUS_PER_NODE="$MA_NUM_GPUS"
NUM_PROCESSES=$(($NGPUS_PER_NODE * $NNODES))

MASTER_PORT="6060"
JOB_ID="1234"

echo "------> system config <------"
echo "VC_WORKER_HOSTS: ${VC_WORKER_HOSTS}"
echo "MASTER_HOST: ${MASTER_HOST}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "NGPUS_PER_NODE: ${NGPUS_PER_NODE}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "${MA_JOB_DIR}"
echo "------>  <------"

# https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/modeldevpt/ptmigr/ptmigr_0022.html
export HCCL_WHITELIST_DISABLE=1

if [[ $NODE_RANK == 0 ]]; then
    EXT_ARGS="--rdzv_conf=is_host=1"
else
    EXT_ARGS=""
fi

# set npu plog env, https://3ms.huawei.com/hi/group/3225441/wiki_6402466.html
ma_vj_name=`echo ${MA_VJ_NAME} | sed 's:ma-job:modelarts-job:g'`
task_name="worker-${VC_TASK_INDEX}"
task_plog_path=${MA_LOG_DIR}/${ma_vj_name}/${task_name}

mkdir -p ${task_plog_path}
export ASCEND_PROCESS_LOG_PATH=${task_plog_path}

echo "plog path: ${ASCEND_PROCESS_LOG_PATH}"

#npu-smi info

export HCCL_CONNECT_TIMEOUT=1800

echo "------> pwd <------"
pwd
echo "------> files <------"
ls


train_workdir=$(cd $(dirname $0); pwd)
echo ${train_workdir}

CMD="python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    $EXT_ARGS \
    --nproc_per_node=$NGPUS_PER_NODE \
    --rdzv_id=$JOB_ID \
    --rdzv_backend=static \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ${train_workdir}/train/finetune.py \
    --base_model=${base_model} \
    --data_path=${train_file} \
    --cache_dir=${cache_dir} \
    --output_dir=${output_dir} \
    --item_embedding_file=${item_embedding_file} \
    --train_config_file=${train_config_file} \
    --distribute_type=${distribute_type}
    "

echo "------> CMD <------"
echo $CMD
$CMD
