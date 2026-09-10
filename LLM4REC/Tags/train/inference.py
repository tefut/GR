import argparse
import json
import os
import time
import torch
from tqdm import tqdm
import numpy as np
import pickle

from torch.cuda.amp import autocast
from dataset import IMCIEPredictDataset, IMCIEPredictCollator
from torch.utils.data import DataLoader
from model import BaseModel, IMCIEModel
from utils import token_ids_to_text, text_to_token_ids, load_file, build_prompt, write_to_file


def run(args):
    base_model = args.base_model
    finetune_model_path = args.finetune_model_path
    cache_dir = args.cache_dir
    data_path = args.data_path
    item_embedding_file = args.item_embedding_file
    train_config_file = args.train_config_file
    output_file = args.output_dir

    with open(train_config_file, 'r', encoding='utf-8') as fin:
        train_config = json.load(fin)
    print(train_config)

    batch_size = train_config.get("infere_batch_size", 256)

    # 构造训练数据
    max_length = train_config.get("max_length", 128)
    item_embeds = pickle.load(open(item_embedding_file, 'rb'))
    input_dim = train_config.get("input_dim", 64)
    output_dim = train_config.get("output_dim", 32)
    embedding_mode = train_config.get("embedding_mode", "mean_pooling")

    data_file = os.path.join(data_path, "game_info_val.json")
    test_data = load_file(data_file)
    finetune_type = train_config.get("finetune_type", 'g')

    test_dataset = IMCIEPredictDataset(test_data)

    ignore_index, device = -100, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_collator = IMCIEPredictCollator(base_model, cache_dir, max_length, ignore_index, device)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)
    tokenizer = data_collator.tokenizer

    # 构造模型
    lora_r = train_config.get("lora_r", 16)
    lora_alpha = train_config.get("lora_alpha", 16)
    lora_dropout = train_config.get("lora_dropout", 0.05)
    lora_target_modules = train_config.get("lora_target_modules", ['query_key_value', 'dense_h_to_4h', 'dense_4h_to_h'])
    lora_config = [lora_r, lora_alpha, lora_dropout, lora_target_modules]

    gen_model = BaseModel(
        base_model=base_model,
        input_dim=input_dim,
        output_dim=output_dim,
        lora_config=lora_config
    )

    cf_model = IMCIEModel(
        base_model=base_model,
        input_dim=input_dim,
        output_dim=output_dim,
        item_embeds=item_embeds,
        lora_config=lora_config,
        save_path=finetune_model_path,
        embedding_mode=embedding_mode
    )

    print(gen_model)
    print(cf_model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen_state_dict = torch.load(os.path.join(finetune_model_path, "/gen_model.pth"))
    gen_new_state_dict = {}
    for key in gen_state_dict:
        for module, vector in gen_state_dict.get(key).items():
            print(f"{key}.{module}")
            gen_new_state_dict[f"{key}.{module}"] = vector

    gen_model_dict = gen_model.state_dict()
    gen_model_dict.update(gen_new_state_dict)
    gen_model.load_state_dict(gen_model_dict)
    del gen_state_dict

    cf_state_dict = torch.load(os.path.join(finetune_model_path, "/cf_model.pth"))
    cf_new_state_dict = {}
    for key in cf_state_dict:
        for module, vector in cf_state_dict.get(key).items():
            print(f"{key}.{module}")
            cf_new_state_dict[f"{key}.{module}"] = vector

    cf_model_dict = cf_model.state_dict()
    cf_model_dict.update(cf_new_state_dict)
    cf_model.load_state_dict(cf_model_dict)
    del cf_state_dict

    generate_text(gen_model, test_data, tokenizer, device, output_file)
    gen_item_embed(cf_model, test_data, tokenizer, device, output_file)


def generate_text(model, test_data, tokenizer, device, output_file):
    model = model.to(device)
    model.eval()

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        _, input_text = build_prompt(entry)

        token_ids = generate(
            model=model,
            tokenizer=tokenizer,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=100,
            context_size=1024,
            eos_id=tokenizer.eos_token_id
        )

        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text) + len("[gMASK]sop ") + 1:].strip()

        test_data[i]["游戏标签"] = response_text

        output_line = test_data[i]["游戏名称"] + "：" + test_data[i]["游戏标签"]
        print(output_line)

    result_file = os.path.join(output_file, 'predict_result.txt')
    with open(result_file, "w", encoding='utf-8') as file:
        json.dump(test_data, file, indent=4)


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            output = model(idx_cond)
        logits = output.logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def gen_item_embed(model, dataloader, tokenizer, device, output_file):
    item_embeddings_output = write_to_file(output_file, 'w')

    print("strat inference ...")
    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs = data['inputs'].to(device)
            inputs_mask = data["inputs_mask"].to(device)
            item_encode_id = data["item_encode_id"].to(device)
            item_ids = data["item_ids"]
            with autocast():
                pooled_logits = model(inputs, inputs_mask, item_encode_id)
            item_embeddings = pooled_logits.cpu().numpy()

            for iid, item_embed in zip(item_ids, item_embeddings):
                item_embed = item_embed / np.linalg.norm(item_embed)
                if np.any(np.isnan(item_embed)) or np.all(item_embed == 0):
                    print(f"error embedding: {item_embed}")
                    continue
                emb = ",".join(map(str, item_embed.tolist()))
                item_embeddings_output.write(str(iid) + "|" + str(emb) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--base_model', required=True, type=str,
                        help='Please specify pretrained language model path')
    parser.add_argument('-d', '--data_path', required=True, type=str,
                        help='Please specify the path of dataset')
    parser.add_argument('-c', '--cache_dir', required=True, type=str,
                        help='Please specify the cache dir')
    parser.add_argument('-t', '--train_config_file', required=True, type=str,
                        help='Please specify the path of train config')
    parser.add_argument('-f', '--finetune_model_path', required=True, type=str)
    parser.add_argument('-o', '--output_dir', required=False, type=str, default="evaluate")

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)
    torch.cuda.empty_cache()

    mtp_train_config_file = "/opt/huawei/schedule-train/algorithm/train.config"
    if os.path.exists(mtp_train_config_file):
        args.train_config_file = mtp_train_config_file

    start_time = time.time()

    data_path = os.path.dirname(args.output_dir)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        print("make dir: ", data_path)

    run(args)
    end_time = time.time()
    print(f"predict time: {end_time - start_time}")


if __name__ == '__main__':
    main()
