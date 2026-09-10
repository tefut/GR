#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "running preprocess.sh"
train_workdir=$(cd $(dirname $0); pwd)
echo ${train_workdir}

python ${train_workdir}/gr_batch_data_preprocess.py \
    --input_path=${input_path} \
    --model_config_file=${model_config_file} \
    --output_path=${output_path} \
    --batch_size=${batch_size} \
    --num_processes=${num_processes} \
