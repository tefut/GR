#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "running run.sh"
train_workdir=$(cd $(dirname $0); pwd)
echo ${train_workdir}

python ${train_workdir}/encode.py \
    --url=${url} \
    --app_id=${app_id} \
    --sign_key=${sign_key} \
    --flow_id=${flow_id} \
    --batch_size=${batch_size} \
    --data_path=${data_path} \
    --output_file=${output_file} \
    --sep=${sep} \
    --max_workers=${max_workers}
