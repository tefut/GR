#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "running infere.sh"
train_workdir=$(cd $(dirname $0); pwd)
echo ${train_workdir}

python ${train_workdir}/train/inference.py \
    --base_model=${base_model} \
    --data_path=${data_path} \
    --cache_dir=${cache_dir} \
    --train_config_file=${train_config_file} \
    --finetune_model_path=${finetune_model_path} \
    --output_dir=${output_dir}
