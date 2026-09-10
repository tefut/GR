#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "running get_more_predict_data.sh"
train_workdir=$(cd $(dirname $0); pwd)
echo ${train_workdir}

python ${train_workdir}/utils/get_more_predict_data.py \
    --play_to_positive_file=${play_to_positive_file} \
    --item_info_file=${item_info_file} \
    --index_to_item_file=${index_to_item_file} \
    --output_file=${output_file}