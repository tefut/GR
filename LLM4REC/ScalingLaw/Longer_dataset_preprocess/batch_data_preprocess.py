#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "Starting batch_data_preprocess_day.sh"

train_workdir=$(cd $(dirname $0); pwd)
echo "Train working directory: ${train_workdir}"

echo "Dataset source path: ${dataset_path_src}"
echo "Dataset target path: ${dataset_path_target}"

date=${period:0:8}
echo "Period: ${period}"
echo "Extracted date: ${date}"

input_path="${dataset_path_src}/${date}/${input_subpath}"
model_config_file="${dataset_path_src}/${date}/${input_config_subpath}"
output_path="${dataset_path_target}/${date}/${output_subpath}"

echo "Input path: ${input_path}"
echo "Model config file path: ${model_config_file}"
echo "Output path: ${output_path}"


echo "Running batch_data_preprocess.py with the following arguments:"
echo "  --input_path=${input_path}"
echo "  --model_config_file=${model_config_file}"
echo "  --output_path=${output_path}"
echo "  --batch_size=${batch_size}"
echo "  --num_processes=${num_processes}"
echo "  --chunk_size=${chunk_size}"
echo "  --mode=${mode}"

python "${train_workdir}/batch_data_preprocess.py" \
    --input_path="${input_path}" \
    --model_config_file="${model_config_file}" \
    --output_path="${output_path}" \
    --batch_size="${batch_size}" \
    --num_processes="${num_processes}" \
    --chunk_size="${chunk_size}" \
    --mode="${mode}"

if [ $? -eq 0 ]; then
    echo "batch_data_preprocess.py executed successfully!"
else
    echo "Error: batch_data_preprocess.py failed to execute."
    exit 1
fi
