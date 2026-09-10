# !/bin/bash
echo -e "\n--> Generative Recommenders on MTP production environment\n"

echo -e "Period: ${period}"
cat train.config

if [ "$step" == "" ]; then
    echo "[ERROR] Invalid step"
    exit 1
fi
steps=$step

output_dir=$output_dir
model_save_dir=$model_save_dir
echo "output dir：$output_dir"
cur_dir=$(dirname $0)

train_workdir=$(cd $(dirname $0); pwd)
echo "train_workdir: ${train_workdir}"

if [ "$period" == "" ]; then
    # 获取最近数据日期
    date=`ls ${data_dir} | grep '^[0-9]\{8\}$' | awk 'BEGIN {max=0} {if($1>max) max=$1} END {print max}'`
    # 必选参数，用以区分测试日期
    period=$(printf "%s-000000" ${date})
else
    date=${period%-*}
fi
echo "period: ${period}"

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
#pip install onnx
#pip install onnxruntime

pip install gin-config absl-py scikit-learn scipy matplotlib numpy apex hypothesis iopath pyarrow
pip install accelerate
end=`date +%s`
runtime=$((end-start))
echo -e "\nTime taken to install python packages: ${runtime} seconds"
echo "====== installed python packages ======"

mxrec_acceleration_ops=$mxrec_acceleration_ops
echo "mxrec_acceleration_ops: $mxrec_acceleration_ops"
if [ -n "$mxrec_acceleration_ops" -a -d "$mxrec_acceleration_ops" ]; then
  echo "====== Install mxrec acclelration ops ======"
  pip install $mxrec_acceleration_ops/torch_plugin/*.whl
  sitepkgs_dir=$(pip show torch |grep -w 'Location:' | awk '{print $2}')
  sed -i '/CUDA = 1/a\    NPU = 2' ${sitepkgs_dir}/fbgemm_gpu/split_table_batched_embeddings_ops_training.py
  for ops_file in `ls -1 $mxrec_acceleration_ops/mxrec_ops/*.run`
  do
    bash $ops_file
  done
  cp -v $mxrec_acceleration_ops/torch_library/*.so $sitepkgs_dir/torch
  echo "====== Install mxrec ops finished ======"
fi

# pip list

# 配置自定义环境变量
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=6000
export HCCL_EXEC_TIMEOUT=6000

export GLOG_logtostderr=1        # 把 glog 日志打到 stderr
export GLOG_v=3                  # 日志级别调到 INFO/DEBUG
export ACL_OP_DEBUG_LEVEL=2      # （如果平台支持）打开 ACL 操作级别的调试

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

: ${data_path_prefix:=""}
data_path="${data_path_prefix}${date}"
echo "plog path: ${ASCEND_PROCESS_LOG_PATH}"
echo "data_path: ${data_dir}${data_path}"

if [[ "$version" == "not_use" ]]; then
    model_save_dir=${data_dir}${data_path}/model
else
    model_save_dir=${data_dir}${data_path}/model/${version}
    mkdir -p ${model_save_dir}

fi
echo "model_save_dir: ${model_save_dir}"


export PYTHONPATH="$PYTHONPATH:./generative_recommenders/"
export NNODES=${NNODES}
# AMP混合精度配置，可通过环境变量覆盖配置文件中的use_amp设置
: ${use_amp:=""}
if [ -n "$use_amp" ]; then
    echo "AMP override from env: use_amp=${use_amp}"
fi
echo "The PYTHONPATH is: ${PYTHONPATH}."

: ${mergeFeatureMaxIndexName:="feature_map.json"}
feature_map_path=${data_dir}${data_path}/config/${mergeFeatureMaxIndexName}
echo "The date_run is: ${date}."
echo "The feature_map_path is: ${feature_map_path}."
mkdir -p "${data_dir}${data_path}"
function train
{
    echo "====== train start ======"
    torchrun --nproc_per_node=${NGPUS_PER_NODE} \
         --nnodes=${NNODES} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
                   ${train_workdir}/main.py --config_file=${train_config_file} \
                  --data_dir=${data_dir}${data_path} \
                  --save_dir=${model_save_dir} \
                  --feature_map_dir=${feature_map_path} \
                  --llm_embedding_path=${data_dir}${data_path}/${llm_embedding_path} \
                  --period=${period}\
                  --is_train=True \
                  --use_amp=${use_amp}
    echo "*** train end ***"
}

if [[ "$steps" =~ "train" ]]; then
    train
fi

function test
{
    echo "====== test start ======"
    torchrun --nproc_per_node=${NGPUS_PER_NODE} \
         --nnodes=${NNODES} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
                   ${train_workdir}/main.py --config_file=${train_config_file} \
                  --data_dir=${data_dir}${data_path} \
                  --save_dir=${model_save_dir} \
                  --feature_map_dir=${feature_map_path} \
                  --llm_embedding_path=${data_dir}${data_path}/${llm_embedding_path} \
                  --period=${period}\
                  --is_train=False \
                  --use_amp=${use_amp}
    echo "*** test end ***"
}

if [[ "$steps" =~ "test" ]]; then
    test
fi

#-----------------------------------------mcp-------------------------------------------

dataLocalPath=${data_dir}${data_path}

function copyFile()
{
    echo "*** mcp $1 is: $2"
    $2
    if [ $? -ne 0 ]
    then
        echo "*** mcp $1 failed ***"
        exit 1
    else
        echo "*** mcp $1 successed ***"
        LOGFILE=$dataLocalPath/log/$date/${1}_success
        LOG_DIR=`dirname $LOGFILE`
        [ ! -d $LOG_DIR ] && mkdir -p $LOG_DIR
        touch $dataLocalPath/log/$date/${1}_success
    fi
}

featureMapLocalDir=${data_dir}${data_path}/config
modelOutputDir=/opt/huawei/schedule-train/output

# featureMap 文件
# demanded files (configurable)
: ${featureMapOldName:="featureMap.txt"}
: ${featureMapOutputName:="featuremap.featuremap"}
: ${columnMapName:="column_feature_map.json"}
: ${mergeMapName:="merge_feature_map.txt"}
: ${train_config:="gr_module_config.json"}
: ${output_train_config:="model_config.json"}
: ${auc_res_file:="result.txt"}
: ${source_config:="source_config.json"}
: ${config_dir:=""}


function mcp
{
    echo "====== mcp start ======"

    # copy model files
    if [[ "$version" == "not_use" ]]; then
        modelSubDirName=model/${date_run}
    else
        modelSubDirName=model/${version}/${date_run}
    fi
    cmdMakeModelDir="mkdir -p ${modelOutputDir}/modelfile"
    cmdMakeConfDir="mkdir -p ${modelOutputDir}/config"
    cmdCopyModel="cp -r ${dataLocalPath}/${modelSubDirName}/modelfile/* ${modelOutputDir}/modelfile"
    echo "*** mcp cmdMakeModelDir is: ${cmdMakeModelDir}"
    $cmdMakeModelDir
    $cmdMakeConfDir
    copyFile "copyModel" "${cmdCopyModel}"

    # copy train_config json file
    configSrc="${dataLocalPath}/${modelSubDirName}/${train_config}"
    if [ -f "$configSrc" ]; then
        cmdCopyConfig="cp ${configSrc} ${modelOutputDir}/${train_config}"
        copyFile "copyConfig" "${cmdCopyConfig}"
        cmdCopyConfig="cp ${dataLocalPath}/${config_dir}/${output_train_config} ${modelOutputDir}/${config_dir}/${output_train_config}"
        copyFile "copyConfig" "${cmdCopyConfig}"
    else
        echo "Config file does not exist: $configSrc"
        exit 1
    fi

    # copy best auc result file
    aucSrc="${dataLocalPath}/${modelSubDirName}/modelfile/${auc_res_file}"
    if [ -f "$aucSrc" ]; then
        cmdCopyRes="cp ${aucSrc} ${modelOutputDir}/${auc_res_file}"
        copyFile "CopyRes" "${cmdCopyRes}"
    else
        echo "AUC result file does not exist: $aucSrc"
    fi

    # copy featuremap files
    declare -A filesMap=(
        ["${featureMapLocalDir}/${columnMapName}"]="${modelOutputDir}/${config_dir}/${columnMapName}"
        ["${featureMapLocalDir}/${mergeFeatureMaxIndexName}"]="${modelOutputDir}/${config_dir}/${mergeFeatureMaxIndexName}"
        ["${featureMapLocalDir}/${mergeMapName}"]="${modelOutputDir}/${config_dir}/${mergeMapName}"
        ["${featureMapLocalDir}/${source_config}"]="${modelOutputDir}/${config_dir}/${source_config}"
    )
    for src in "${!filesMap[@]}"; do
        dst="${filesMap[$src]}"
        if [ -f "$src" ]; then
            cmd="cp $src $dst"
            copyFile "file $src" "$cmd"
        else
            echo "Source file does not exist, skipping: $src"
        fi
    done

    echo "*** mcp end ***"
}

if [[ "$steps" =~ "mcp" && "${NODE_RANK}" == "0" ]]; then
    mcp
fi
