#!/bin/bash
date=${period%-*}
config_file=/opt/huawei/schedule-train/algorithm/train.config
featuremap_dir=$data_dir/
data_dir=$data_dir/train_data/
save_dir=${PYTHONPATH}/local_save_dir/
emb_dir=$out_dir/local_emb_dir/
curr_date=${period}

echo -e "\n--> Generative Recommenders on ROMA environment\n"

echo -e "Period: ${period}"

echo "Current Traing Period is ${date}"

echo "Data Directory is ${data_dir}"

echo "Feature map Directory is ${featuremap_dir}"

export RANK=0
export WORLD_SIZE=2
export LOCAL_RANK=0

torchrun --nproc_per_node=2 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=127.0.0.1 \
         --master_port=12348 \
                  ${PYTHONPATH}/python/LLM4REC/ScalingLaw/embedding_main.py --config_file=$config_file \
                  --data_dir=$data_dir \
                  --save_dir=$save_dir \
                  --emb_dir=$emb_dir \
                  --period=$curr_date \
                  --is_train=True \
                  --save_user_emb=True \
                  --feature_map_dir=$featuremap_dir/watch_sequence_featuremap.csv \
                  --tensorboard_log_dir=$save_dir/tensorboard
