#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

echo "running main_kar_full.sh"
train_workdir=$(cd $(dirname $0); pwd)
prompt_dir=${save_path}/prompt
generate_dir='generate'
encoding_dir='encoding'

python -u ${train_workdir}/step1_music_prompt_generator.py --user_info_path=${user_info_path} \
        --user_click_info_path=${user_click_info_path} --item_info_path=${item_info_path} \
        --save_path=${prompt_dir} --sep=${sep}  --user_prefix=${user_prefix} \
        --user_count_train_threshold=${user_count_train_threshold} \
        --user_count_his_threshold=${user_count_his_threshold} \
        --max_user_seq_len=${max_user_seq_len} \
        --avg_user_seq_len=${avg_user_seq_len}
echo "Finish prompt generation for full data"
echo "#" * 80

python -u ${train_workdir}/step2_llm_generate_response.py --model_path=${model_path} --model_type=${generate_mode_type} \
        --input_path=${prompt_dir}/${user_prefix} \
        --output_path=${save_path}/${generate_dir}/${generate_mode_type} --sep=${sep}
echo "Finish user knowledge generation"
echo "#" * 80

python -u ${train_workdir}/step3_llm_encoding_generate_pca.py --model_path=${model_path} --model_type=${encoding_model_type} \
        --input_path=${save_path}/${generate_dir}/${generate_mode_type} \
        --output_path=${save_path}/${encoding_dir}/${encoding_model_type} \
        --sep=${sep} --pca_batch_size=${pca_batch_size} --pca_target_dim=${pca_target_dim} \
        --llm_batch_size=${llm_batch_size}
echo "Finish encoding & pca generation for user liber"
echo "#" * 80


python -u ${train_workdir}/step4_get_final_embedding.py --user_info_path=${user_info_path} \
        --item_info_path=${item_info_path} --llm_input_path=${save_path}/${encoding_dir}/${encoding_model_type} \
        --pca_target_dim=${pca_target_dim} --sep=${sep}
echo "Finish combine all user embedding & generate group id"
echo "#" * 80



exit $?
