#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "running get_cross_domain_train_data.sh"
train_workdir=$(cd $(dirname $0); pwd)
echo ${train_workdir}

echo "-----------------------------------------------------step 1-----------------------------------------------------------"
python ${train_workdir}/cross_domain/get_domain_item_prompt.py \
    --item_file_path=${item_file_path} \
    --item_prompt_path=${item_prompt_path} \
    --prompt_config_file=${prompt_config_file}

echo "-----------------------------------------------------step 2-----------------------------------------------------------"
python ${train_workdir}/cross_domain/get_train_data.py \
    --item_sample_path=${item_sample_path} \
    --item_embedding_path=${item_embedding_path} \
    --item_prompt_path=${item_prompt_path} \
    --save_path=${save_path}
