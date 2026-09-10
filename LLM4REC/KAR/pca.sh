#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "running run.sh"
train_workdir=$(cd $(dirname $0); pwd)

python -u ${train_workdir}/incremental_pca.py \
        --input_path=${input_path} \
        --output_path=${output_path} \
        --sep=${sep} \
        --pca_batch_size=${pca_batch_size} \
        --pca_target_dim=${pca_target_dim} \
        --precision_float_number=${precision_float_number}
echo "Finish encoding & pca generation for item"
echo "#"*80

exit $?
