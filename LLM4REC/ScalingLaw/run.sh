#!/bin/bash
echo -e "\n--> Generative Recommenders on MTP production environment\n"

echo -e "Period: ${period}"
cat train.config

output_dir=$output_dir
model_save_dir=$model_save_dir
echo "output dir：$output_dir"
cur_dir=$(dirname $0)

train_workdir=$(cd $(dirname $0); pwd)
echo ${train_workdir}

# 处理所有必选参数
if [ "$scenName" == "" ]; then
    echo "[ERROR] Invalid scenName"
    exit 1
fi
export scene_name=$scenName

if [ "$modelId" == "" ]; then
    echo "[ERROR] Invalid modelId"
    exit 1
fi
model_id=$modelId

date=${period%-*}

echo "Current Traing Period is ${date}"

if [ "$version" == "" ]; then
    echo "[ERROR] Invalid version"
    exit 1
fi
version=$version

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


# 指定要检查的文件夹
directory="$output_dir/python_package"

# 检查文件夹是否存在
if [ -d "$directory" ]; then
    echo "python package dir：$directory"
    # 在这里可以执行其他操作
    pip install $output_dir/python_package/*whl
else
    echo "python package dir：$directory"
fi

# 安装 ONNX 导出所需包
pip install onnx
pip install onnxruntime

pip install gin-config absl-py scikit-learn scipy matplotlib numpy apex hypothesis iopath pyarrow
pip install accelerate

# pip install mindstudio-probe

end=`date +%s`
runtime=$((end-start))
echo -e "\nTime taken to install python packages: ${runtime} seconds"
echo "====== installed python packages ======"


for ops_file in `ls -1 $cur_dir/ops/*.run`
do
    bash $ops_file
done

pip list

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

MASTER_PORT="12345"
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
echo "${data_dir}/${scenName}/${modelId}/data/${date}/train"

echo "save_dir: ${model_save_dir}"


export PYTHONPATH="$PYTHONPATH:./generative_recommenders/"
echo "The PYTHONPATH is: ${PYTHONPATH}." 
export NNODES=${NNODES}
torchrun --nproc_per_node=${NGPUS_PER_NODE} \
         --nnodes=${NNODES} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
                   ${train_workdir}/main.py --config_file=${train_config_file} \
                  --data_dir=${data_dir}/${data_file_path} \
                  --save_dir=${model_save_dir} \
                  --feature_map_dir=${data_dir}/${feature_map_file_path} \
                  --LLM_embedding_dir=${data_dir} \
                  --period=${period}\
                  --is_train=${is_train}
