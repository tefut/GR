#!/bin/bash
echo -e "\n Starting generating preprocess\n"

if [ "$period" == "" ]; then
    # 获取最近数据日期
    date=`ls ${data_dir} | grep '^[0-9]\{8\}$' | awk 'BEGIN {max=0} {if($1>max) max=$1} END {print max}'`
    # 必选参数，用以区分测试日期
    period=$(printf "%s-000000" ${date})
else
    date=${period%-*}
fi
echo "period: ${period}"

train_workdir=$(cd $(dirname $0); pwd)
echo "train_workdir: ${train_workdir}"
export PYTHONPATH="${train_workdir}:${PYTHONPATH}" 


INPUT_DIR=${INPUT_DIR}/${date}/${out}
OUTPUT_DIR=${OUTPUT_DIR}/${date}/${out_opt}

if [[ "$step" =~ "preprocess" ]]; then
    python ${train_workdir}/${code_dir}/preprocess.py \
        --INPUT_DIR ${INPUT_DIR} \
        --OUTPUT_DIR ${OUTPUT_DIR} \
        --config_file ${config_file} 
fi

