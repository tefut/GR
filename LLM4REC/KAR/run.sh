#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "running run.sh"
train_workdir=$(cd $(dirname $0); pwd)

prompt_save_path=${output_path}/prompt

python -u ${train_workdir}/prompt_generator.py \
        --info_path=${info_path} \
        --prompt_save_path=${prompt_save_path} \
        --prompt_str=${prompt_str} \
        --orc_file=${orc_file} \
        --all_columns=${all_columns} \
        --sep=${sep} \
        --num_lines=${num_lines}
echo "Finish prompt generation for full data"
echo "#"*80

python -u ${train_workdir}/llm_encoding_pca.py \
        --model_path=${model_path} \
        --model_type=${model_type} \
        --input_path=${prompt_save_path} \
        --output_path=${output_path} \
        --sep=${sep} \
        --pca_batch_size=${pca_batch_size} \
        --pca_target_dim=${pca_target_dim} \
        --llm_batch_size=${llm_batch_size} \
        --max_length=${max_length} \
        --precision_float_number=${precision_float_number} \
        --mode=${mode}
echo "Finish encoding & pca generation for item"
echo "#"*80

exit $?
