#!/bin/bash
echo -e "\n--> Generative Recommenders on ROMA environment\n"

echo -e "Period: ${period}"

date=${period%-*}
echo "Current Traing Period is ${date}"

export RANK=0
export WORLD_SIZE=2
export LOCAL_RANK=0

torchrun --nproc_per_node=2 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=127.0.0.1 \
         --master_port=12348 \
                   embedding_main.py --config_file="/home/l00856372/gr-gpu/RecAlgorithmBase-yq-dev/tests/config/test_config.yaml" \
                  --data_dir="/home/l00856372/gr-gpu/amazon_books/" \
                  --save_dir="/home/l00856372/gr-gpu/modelfile/" \
                  --period="20230914-000000" \
                  --is_train=True \
                  --save_user_emb=False \
                  --tensorboard_log_dir="/home/l00856372//gr-gpu/runs/exps_amzn_books"
