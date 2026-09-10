import json
import os
import stat
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.trainer_utils import get_last_checkpoint
from safetensors.torch import load_file

from dataset import PredictDataset, PredictCollator
from trainer import load_model


def write_to_file(save_file, mode='w'):
    flags = os.O_WRONLY | os.O_CREAT
    stats = stat.S_IWUSR | stat.S_IRUSR
    file_hander = os.fdopen(os.open(save_file, flags, stats), mode)
    return file_hander


def load_predict_dataset(dataset_args):
    '''加载训练数据集和验证数据集'''
    max_length = dataset_args.get("max_length", 128)
    base_model = dataset_args.get("pretrained_model_path")
    cache_dir = dataset_args.get("cache_dir")

    predict_file = dataset_args.get("predict_data_path", "")
    predict_dataset = PredictDataset(predict_file)
    predict_data_collator = PredictCollator(base_model, cache_dir, max_length)
    return predict_dataset, predict_data_collator


def predict(dataset_args, model_args, predict_args):
    checkpoint_path = predict_args.get("checkpoint_path")
    output_file = predict_args.get("output_file")
    predict_batch_size = predict_args.get("predict_batch_size")

    model = load_model(model_args)
    last_checkpoint_path = get_last_checkpoint(checkpoint_path)
    if last_checkpoint_path is None:
        raise ValueError(f"No valid checkpoint found in output directory ({checkpoint_path})")
    state_dict = load_file(os.path.join(last_checkpoint_path, "model.safetensors"))
    model.load_state_dict(state_dict)
    del state_dict

    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print("device", device)

    predict_dataset, predict_data_collator = load_predict_dataset(dataset_args)
    dataloader = DataLoader(predict_dataset, batch_size=predict_batch_size, collate_fn=predict_data_collator)

    item_embeddings_output = write_to_file(output_file, 'w')

    print("strat inference ...")
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_ids = data['input_ids'].to(device)
            input_mask = data["input_mask"].to(device)
            item_ids = data["item_ids"].to(device)
            song_ids = data["song_ids"]

            pooled_logits = model(input_ids, input_mask, item_ids)

            item_embeddings = pooled_logits.float().cpu().numpy()

            for sid, item_embed in zip(song_ids, item_embeddings):
                item_embed = item_embed / np.linalg.norm(item_embed)
                emb = ",".join(map(str, item_embed.tolist()))
                item_embeddings_output.write(str(sid) + "|" + str(emb) + "\n")


def main():
    train_config_file = "/opt/huawei/schedule-train/algorithm/train.config"
    with open(train_config_file, 'r', encoding='utf-8') as fin:
        train_config = json.load(fin)
    print(train_config)

    dataset_args = train_config.get("dataset_configs")
    model_args = train_config.get("model_configs")
    predict_args = train_config.get("predict_configs")

    predict(dataset_args, model_args, predict_args)


if __name__ == '__main__':
    main()
