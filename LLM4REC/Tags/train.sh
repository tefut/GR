#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "running train.sh"
train_workdir=$(cd $(dirname $0); pwd)
echo ${train_workdir}

python ${train_workdir}/train/finetune.py \
    --base_model=${base_model} \
    --data_path=${train_file} \
    --cache_dir=${cache_dir} \
    --output_dir=${output_dir} \
    --train_config_file=${train_config_file} \
    --distribute_type=${distribute_type}
