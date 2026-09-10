#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "running import_recall_item.sh"
train_workdir=$(cd $(dirname $0); pwd)
echo ${train_workdir}

python ${train_workdir}/utils/import_recall_item.py \
    --input_file=${input_file} \
    --output_file=${output_file} \
    --sep=${sep} \
    --top_k=${top_k} \
    --top_p=${top_p}