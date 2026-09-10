import argparse
import json
import os
import pickle
import stat
import time

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PredictDataset, PredictCollator
from model import SimCIE


def write_to_file(save_file, mode='w'):
    flags = os.O_WRONLY | os.O_CREAT
    stats = stat.S_IWUSR | stat.S_IRUSR
    file_hander = os.fdopen(os.open(save_file, flags, stats), mode)
    return file_hander


def predict(args):
    base_model = args.base_model
    finetune_model_path = args.finetune_model_path
    cache_dir = args.cache_dir
    data_path = args.data_path
    item_embedding_file = args.item_embedding_file
    train_config_file = args.train_config_file
    output_file = args.output_file

    with open(train_config_file, 'r', encoding='utf-8') as fin:
        train_config = json.load(fin)
    print(train_config)

    batch_size = train_config.get("infere_batch_size", 256)

    # 构造训练数据
    max_length = train_config.get("max_length", 128)
    dataset = PredictDataset(data_path)
    item_embeds = pickle.load(open(item_embedding_file, 'rb'))
    input_dim = train_config.get("input_dim", 64)
    output_dim = train_config.get("output_dim", 32)
    embedding_mode = train_config.get("embedding_mode", "mean_pooling")
    data_collator = PredictCollator(base_model, cache_dir, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    # 构造模型
    lora_r = train_config.get("lora_r", 16)
    lora_alpha = train_config.get("lora_alpha", 16)
    lora_dropout = train_config.get("lora_dropout", 0.05)
    lora_target_modules = train_config.get("lora_target_modules", ['query_key_value', 'dense_h_to_4h', 'dense_4h_to_h'])
    lora_config = [lora_r, lora_alpha, lora_dropout, lora_target_modules]
    model = SimCIE(
        base_model=base_model,
        input_dim=input_dim,
        output_dim=output_dim,
        item_embeds=item_embeds,
        lora_config=lora_config,
        save_path=finetune_model_path,
        embedding_mode=embedding_mode
    )

    # 加载模型参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(os.path.join(finetune_model_path, "adapter.pth"))
    new_state_dict = {}
    for key in state_dict:
        for module, vector in state_dict.get(key).items():
            print(f"{key}.{module}")
            new_state_dict[f"{key}.{module}"] = vector

    model_dict = model.state_dict()
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    del state_dict

    model = model.to(device)
    model.eval()

    item_embeddings_output = write_to_file(output_file, 'w')

    total_index_ids, total_song_ids, total_item_embeddings = [], [], []
    print("strat inference ...")
    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs = data['inputs'].to(device)
            inputs_mask = data["inputs_mask"].to(device)
            item_ids = data["item_ids"].to(device)
            song_ids = data["song_ids"]
            with autocast():
                pooled_logits = model(inputs, inputs_mask, item_ids)
            item_embeddings = pooled_logits.cpu().numpy()
            total_index_ids.extend(item_ids.cpu().numpy().tolist())
            total_song_ids.extend(song_ids)
            total_item_embeddings.extend(item_embeddings)

            for sid, item_embed in zip(song_ids, item_embeddings):
                item_embed = item_embed / np.linalg.norm(item_embed)
                if np.any(np.isnan(item_embed)) or np.all(item_embed == 0):
                    print(f"error embedding: {item_embed}")
                    continue
                emb = ",".join(map(str, item_embed.tolist()))
                item_embeddings_output.write(str(sid) + "|" + str(emb) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--base_model', required=True, type=str,
                        help='Please specify pretrained language model path')
    parser.add_argument('-d', '--data_path', required=True, type=str,
                        help='Please specify the path of dataset')
    parser.add_argument('-c', '--cache_dir', required=True, type=str,
                        help='Please specify the cache dir')
    parser.add_argument('-i', '--item_embedding_file', required=True, type=str,
                        help='Please specify the path of item embedding file')
    parser.add_argument('-t', '--train_config_file', required=True, type=str,
                        help='Please specify the path of train config')
    parser.add_argument('-f', '--finetune_model_path', required=True, type=str)
    parser.add_argument('-o', '--output_file', required=False, type=str, default="evaluate")

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)
    torch.cuda.empty_cache()

    mtp_train_config_file = "/opt/huawei/schedule-train/algorithm/train.config"
    if os.path.exists(mtp_train_config_file):
        args.train_config_file = mtp_train_config_file

    start_time = time.time()

    data_path = os.path.dirname(args.output_file)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        print("make dir: ", data_path)

    predict(args)
    end_time = time.time()
    print(f"predict time: {end_time - start_time}")


if __name__ == '__main__':
    main()
