#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "running get_word2vec_train_data.sh"
train_workdir=$(cd $(dirname $0); pwd)
echo ${train_workdir}

python ${train_workdir}/utils/get_word2vec_train_data.py \
    --data_path=${data_path} \
    --output_file=${output_file} \
    --sep=${sep} \
    --target_column=${target_column}  \
    --min_item_num=${min_item_num}