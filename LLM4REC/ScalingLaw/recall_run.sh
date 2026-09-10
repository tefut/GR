#!/bin/bash
echo -e "\n--> Generative Recommenders on MTP production environment\n"

train_workdir=$(cd $(dirname $0); pwd)
echo ${train_workdir}

free -h
npu-smi info

# 配置CANN相关环境变量
echo -e "\n====== 配置CANN相关环境变量 ======"
CANN_INSTALL_PATH_CONF='/etc/Ascend/ascend_cann_install.info'
sudo cat /etc/Ascend/ascend_cann_install.info
DEFAULT_CANN_INSTALL_PATH="/usr/local/Ascend/"
CANN_INSTALL_PATH="/usr/local/Ascend/"

if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ];then
  cat ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
  source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
  cat ${CANN_INSTALL_PATH}/nnae/set_env.sh
  source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi
echo -e "\n====== CANN相关环境变量配置完成 ======"

echo -e "\n====== installing python packages ======"
start=`date +%s`

hccl_info --version
dpkg -l | grep ascend-toolkit
mpirun --allow-run-as-root -n 8 hccl_test --comm_test


# 安装 ONNX 导出所需包
pip install onnx
pip install onnxruntime

pip install gin-config absl-py scikit-learn scipy matplotlib numpy apex hypothesis iopath pyarrow
end=`date +%s`
runtime=$((end-start))
echo -e "\nTime taken to install python packages: ${runtime} seconds"


# 配置自定义环境变量
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=6000
export HCCL_EXEC_TIMEOUT=6000

# log
export ASCEND_SLOG_PRINT_TO_STDOUT=0   # 日志打屏, 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       # 日志级别常用 1 INFO级别; 3 ERROR级别
export ASCEND_GLOBAL_EVENT_ENABLE=0    # 默认不使能event日志信息
export ASCEND_LAUNCH_BLOCKING=0        # 默认不开启算子下发同步，影响训练性能; 开启后每执行完一个算子会做一次流同步

# 系统默认环境变量，不建议修改
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NNODES="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
NGPUS_PER_NODE="$MA_NUM_GPUS"
NUM_PROCESSES=$(($NGPUS_PER_NODE * $NNODES))

MASTER_PORT="13456"
JOB_ID="1234"

echo "------> system config <------"
echo "VC_WORKER_HOSTS: ${VC_WORKER_HOSTS}"
echo "MASTER_HOST: ${MASTER_HOST}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "NGPUS_PER_NODE: ${NGPUS_PER_NODE}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "${MA_JOB_DIR}"
echo "------>  <------"

export HCCL_PRIORITY=1
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


torchrun --nproc_per_node=${NGPUS_PER_NODE} \
         --nnodes=${NNODES} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${train_workdir}/recall_main.py --device "npu"
