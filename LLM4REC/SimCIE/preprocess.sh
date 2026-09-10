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

echo "-----------------------------------------------------step 1-----------------------------------------------------------"
python ${train_workdir}/utils/get_item_info.py \
    --item_file_path=${item_file_path} \
    --item_info_file=${item_info_file}

echo "-----------------------------------------------------step 2-----------------------------------------------------------"
python ${train_workdir}/utils/kmean.py \
    --pretrained_item_embedding_file=${pretrained_item_embedding_file} \
    --data_path=${data_path} \
    --norm=${norm} \
    --n_clusters=${n_clusters}

echo "-----------------------------------------------------step 3-----------------------------------------------------------"
python ${train_workdir}/utils/get_train_data.py \
    --data_path=${data_path} \
    --item_info_file=${item_info_file} \
    --sim_score_file=${sim_score_file} \
    --mode=${mode} \
    --number=${number} \
    --top_k=${top_k} \
    --top_p=${top_p}