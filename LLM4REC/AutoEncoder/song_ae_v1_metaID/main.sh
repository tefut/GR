#!/bin/bash

echo "running train.sh"

train_workdir=$(cd $(dirname $0); pwd)

echo ${train_workdir}

python ${train_workdir}/main.py \

    --data_path=${data_path} \

    --output_dir=${output_dir} \

    --keep_embd_dir=${keep_embd_dir} \

    --train_config_file=${train_config_file} \

    --distribute_type=${distribute_type}