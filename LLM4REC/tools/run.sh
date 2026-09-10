#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "running run.sh"
train_workdir=$(cd $(dirname $0); pwd)


python -u ${train_workdir}/read_tfrecord_data.py \
        --number_of_samples=${number_of_samples} \
        --data_path=${data_path} \
        --model_config_file=${model_config_file}

exit $?
