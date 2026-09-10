import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoTokenizer

from dataset import SequentialDataset, SequentialCollator
from metrics import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, getLabel
from model import LLM4Rec


def train(local_rank, args):
    base_model = args.base_model
    data_path = args.data_path
    cache_dir = args.cache_dir
    output_dir = args.output_dir
    item_embedding_file = args.item_embedding_file
    train_config_file = args.train_config_file
    distribute_type = args.distribute_type

    with open(train_config_file, 'r', encoding='utf-8') as fin:
        train_config = json.load(fin)
    print(train_config)

    # 构造训练数据
    max_user_seq_len = train_config.get("max_user_seq_len", 30)
    dataset = SequentialDataset(data_path, max_user_seq_len)
    item_embed = pickle.load(open(item_embedding_file, 'rb'))
    num_items = len(item_embed) + 1
    data_collator = SequentialCollator()

    # 构造模型
    model = LLM4Rec(
        base_model=base_model,
        input_dim=64,
        output_dim=num_items,
        lora_r=train_config.get("lora_r", 16),
        lora_alpha=train_config.get("lora_alpha", 16),
        lora_dropout=train_config.get("lora_dropout", 0.05),
        lora_target_modules=train_config.get("lora_target_modules", ["query_key_value"]),
        input_embeds=item_embed,
    )

    # 设置并行模式
    training_type = "general"
    if train_config.get("mixed_precision", True):
        training_type = "mixed_precision"
        batch_size = train_config.get("batch_size", 56)
    else:
        batch_size = 12

    distribute_type = distribute_type.lower()

    if distribute_type == "dp":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallel(model)
        training_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    elif distribute_type == "ddp":
        torch.cuda.set_device(local_rank)
        args.rank = local_rank
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.num_gpus, rank=args.rank)
        device = torch.device("cuda", local_rank)
        model.to(device)
        model = DDP(model, device_ids=[local_rank])

        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        training_dataloader = DataLoader(dataset, batch_size=batch_size,
                                         collate_fn=data_collator, sampler=train_sampler)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        training_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    # 是否进行混合精度训练
    if training_type == "mixed_precision":
        scaler = GradScaler()
    else:
        model.float()

    # 设置可训练参数，并构造优化器
    for name, param in model.named_parameters():
        for pname in ["lora", "input_proj", "score", 'output_proj']:
            param.requires_grad = True if pname in name else False
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(parameters,
                      lr=train_config.get("learning_rate", 0.0003),
                      betas=(0.8, 0.999), weight_decay=3e-7)

    prompt = get_prompt(base_model, cache_dir, device)

    # 开始训练
    print("strat training ...")
    num_epoch = train_config.get("num_epochs", 1)
    for epoch in range(num_epoch):
        running_loss = 0.0
        optimizer.zero_grad()
        if distribute_type == "ddp":
            training_dataloader.sampler.set_epoch(epoch)
        for index, data in enumerate(training_dataloader):
            optimizer.zero_grad()
            inputs = data['inputs'].to(device)
            inputs_mask = data["inputs_mask"].to(device)
            labels = data['labels'].to(device)

            if training_type == "mixed_precision":
                with autocast():
                    _, pooled_logits = model(inputs, inputs_mask, *prompt)
                    loss = F.cross_entropy(pooled_logits, labels.view(-1)).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=20)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, pooled_logits = model(inputs, inputs_mask, *prompt)
                loss = F.cross_entropy(pooled_logits, labels.view(-1)).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=20)  # 梯度裁剪
                optimizer.step()

            running_loss += loss.item()
            log_interval = train_config.get("log_interval", 1000)
            if index % log_interval == (log_interval - 1):
                print(f'[{epoch + 1} {index + 1} loss: {running_loss / log_interval}]')
                running_loss = 0.0

        # 保存模型
        if distribute_type.endswith("dp"):
            if dist.get_rank() == 0:
                save_model(model.module, output_dir, epoch)
        else:
            save_model(model, output_dir, epoch)

        # 模型评估
        evaluate(dataset, model, prompt, device)


def get_prompt(base_model, cache_dir, device):
    template = {
        "prompt_input": f"基于用户的听歌历史，预测用户下一个点击的歌曲。用户的听歌历史如下：\n",
        "response_split": f"下一个点击歌曲为：\n"
    }
    ins = template["prompt_input"]
    res = template["response_split"]
    instruction_text = [ins, res]

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True,
                                              cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    instruct_ids, instruct_mask, _ = tokenizer(instruction_text[0],
                                               truncation=True, padding=False,
                                               return_tensors='pt',
                                               add_special_tokens=False).values()
    response_ids, response_mask, _ = tokenizer(instruction_text[1],
                                               truncation=True, padding=False,
                                               return_tensors='pt',
                                               add_special_tokens=False).values()
    instruct_ids = instruct_ids.to(device)
    instruct_mask = instruct_mask.to(device)
    response_ids = response_ids.to(device)
    response_mask = response_mask.to(device)
    prompt = [instruct_ids, instruct_mask, response_ids, response_mask]
    return prompt


def evaluate(dataset, model, prompt, device, training_type="mixed_precision"):
    # 模型评估
    topk = [1, 5, 10, 20, 100]
    results = {'Precision': np.zeros(len(topk)),
               'Recall': np.zeros(len(topk)),
               'MRR': np.zeros(len(topk)),
               'MAP': np.zeros(len(topk)),
               'NDCG': np.zeros(len(topk))}

    model.eval()
    testData = dataset.testData
    allPos = dataset.allPos

    index = 0
    for u in tqdm(allPos):
        if u not in testData or len(testData[u]) == 0:
            continue
        index += 1
        selected_items = [[testData[u][1]] + dataset.allPos[u]]
        groundTruth = [[0]]
        inputs = torch.LongTensor(testData[u][0]).to(device).unsqueeze(0)
        inputs_mask = torch.ones(inputs.shape).to(device)
        if training_type == "mixed_precision":
            with autocast():
                _, ratings = model(inputs, inputs_mask, *prompt)
        else:
            _, ratings = model(inputs, inputs_mask, *prompt)
        ratings = ratings[[[[k] * len(selected_items[0]) for k in range(len(ratings))], selected_items]]

        _, ratings_K = torch.topk(ratings, k=topk[-1])
        ratings_K = ratings_K.cpu().numpy()
        r = getLabel(groundTruth, ratings_K)

        for j, k in enumerate(topk):
            pre, rec = RecallPrecision_atK(groundTruth, r, k)
            mrr = MRR_atK(groundTruth, r, k)
            map_score = MAP_atK(groundTruth, r, k)
            ndcg = NDCG_atK(groundTruth, r, k)
            results['Precision'][j] += pre
            results['Recall'][j] += rec
            results['MRR'][j] += mrr
            results['MAP'][j] += map_score
            results['NDCG'][j] += ndcg

    for key in results.keys():
        results[key] /= float(len(allPos))
    print(f'Evaluation for User: \n')
    for j, k in enumerate(topk):
        print(f'Precision@{k}: {results["Precision"][j]} \n '
              f'Recall@{k}: {results["Recall"][j]} \n '
              f'MRR@{k}: {results["MRR"][j]} \n '
              f'MAP@{k}: {results["MAP"][j]} \n '
              f'NDCG@{k}: {results["NDCG"][j]} \n')


def save_model(model, output_dir, epoch):
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sub_path = os.path.join(output_dir, f"epoch_{str(epoch)}")
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    model.model.save_pretrained(sub_path)
    model_path = os.path.join(sub_path, "adapter.pth")
    input_proj, output_proj, score = model.input_proj.state_dict(), \
                                     model.output_proj.state_dict(), \
                                     model.score.state_dict()
    torch.save({'input_proj': input_proj, 'output_proj': output_proj, 'score': score}, model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--base_model', required=True, type=str,
                        help='Please specify pretrained language model path')
    parser.add_argument('-d', '--data_path', required=True, type=str,
                        help='Please specify the path of dataset')
    parser.add_argument('-c', '--cache_dir', required=True, type=str,
                        help='Please specify the cache dir')
    parser.add_argument('-o', '--output_dir', required=True, type=str,
                        help='Please specify the path of output')
    parser.add_argument('-i', '--item_embedding_file', required=True, type=str,
                        help='Please specify the path of item embedding file')
    parser.add_argument('-u', '--user_embedding_file', required=False, type=str, default="",
                        help='Please specify the path of user embedding file')
    parser.add_argument('-t', '--train_config_file', required=True, type=str,
                        help='Please specify the path of train config')
    parser.add_argument('-r', '--distribute_type', default="general", type=str)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)
    torch.cuda.empty_cache()

    start_time = time.time()
    args.num_gpus = torch.cuda.device_count()

    if args.num_gpus == 1:
        args.distribute_type = "general"
        train(local_rank=0, args=args)
    elif args.num_gpus > 1 and args.distribute_type.lower() == "ddp":
        torch.multiprocessing.set_start_method("spawn")
        port_id = 10000 + np.random.randint(0, 1000)
        args.dist_url = "tcp://127.0.0.1:" + str(port_id)
        print(args)
        mp.spawn(train, nprocs=args.num_gpus, args=(args,))
    else:
        local_rank = os.environ.get("LOCAL_RANK", 0)
        train(local_rank=local_rank, args=args)
    end_time = time.time()
    print(f"training time: {end_time - start_time}")


if __name__ == '__main__':
    main()
