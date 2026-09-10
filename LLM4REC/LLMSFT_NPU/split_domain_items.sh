#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "running split_domain_items.sh"
train_workdir=$(cd $(dirname $0); pwd)
echo ${train_workdir}


python ${train_workdir}/cross_domain/split_domain_items.py \
    --item_embedding_file=${item_embedding_file}
