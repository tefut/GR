import argparse
import json
import os
import pickle
import time
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AdamW

from dataset import CLDataset, CLCollator
from model import SimCIE


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
    max_length = train_config.get("max_length", 128)
    train_file = os.path.join(data_path, "train.csv")
    eval_file = os.path.join(data_path, "eval.csv")
    training_dataset = CLDataset(train_file)
    eval_dataset = CLDataset(eval_file)
    item_embeds = pickle.load(open(item_embedding_file, 'rb'))
    input_dim = train_config.get("input_dim", 64)
    output_dim = train_config.get("output_dim", 32)
    embedding_mode = train_config.get("embedding_mode", "mean_pooling")  # pooled_output,max_pooling,mean_pooling
    data_collator = CLCollator(base_model, cache_dir, max_length)

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
        embedding_mode=embedding_mode
    )

    # 设置并行模式
    training_type = "general"
    if train_config.get("mixed_precision", True):
        training_type = "mixed_precision"
        batch_size = train_config.get("batch_size", 12)
    else:
        batch_size = train_config.get("batch_size", 4)

    distribute_type = distribute_type.lower()

    print("distribute_type:", distribute_type)
    if distribute_type == "dp":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallel(model)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, collate_fn=data_collator,
                                         shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)
    elif distribute_type == "ddp":
        torch.cuda.set_device(local_rank)
        args.rank = local_rank
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.num_gpus, rank=args.rank)
        device = torch.device("cuda", local_rank)
        model.to(device)
        model = DDP(model, device_ids=[local_rank])

        train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=True)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size,
                                         collate_fn=data_collator, sampler=train_sampler)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, collate_fn=data_collator,
                                         shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

    # 是否进行混合精度训练
    if training_type == "mixed_precision":
        scaler = GradScaler()
    else:
        model.float()

    # 设置可训练参数，并构造优化器
    for name, param in model.named_parameters():
        if "lora" in name or "input_proj" in name or "output_proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(parameters,
                      lr=train_config.get("learning_rate", 0.00003),
                      betas=(0.8, 0.999), weight_decay=3e-7)

    # 开始训练
    print("start training ...")
    num_epoch = train_config.get("num_epochs", 1)
    best_val_loss = float('inf')
    for epoch in range(num_epoch):
        running_loss = 0.0
        if distribute_type == "ddp":
            training_dataloader.sampler.set_epoch(epoch)
        for index, data in enumerate(training_dataloader):
            inputs = data['inputs'].to(device)
            inputs_mask = data["inputs_mask"].to(device)
            item_ids = data["item_ids"].to(device)

            if training_type == "mixed_precision":
                with autocast():
                    pooled_logits = model(inputs, inputs_mask, item_ids)
                    if distribute_type == "ddp":
                        loss = model.module.unsup_loss(pooled_logits)
                    else:
                        loss = model.unsup_loss(pooled_logits)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=20)
                scaler.step(optimizer)
                scaler.update()
            else:
                pooled_logits = model(inputs, inputs_mask, item_ids)
                if distribute_type == "ddp":
                    loss = model.module.unsup_loss(pooled_logits)
                else:
                    loss = model.unsup_loss(pooled_logits)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=20)  # 梯度裁剪
                optimizer.step()
            running_loss += loss.item()
            log_interval = train_config.get("log_interval", 1000)
            if index % log_interval == (log_interval - 1):
                print(f'[Epoch:{epoch + 1} Step:{index + 1} loss: {running_loss / log_interval}]')
                running_loss = 0.0
        val_loss = eval_model(model, eval_dataloader, device, training_type, distribute_type)
        if val_loss > best_val_loss:
            continue
        best_val_loss = val_loss
        # 保存模型
        if distribute_type.endswith("dp"):
            if dist.get_rank() == 0:
                save_model(model.module, output_dir, epoch)
        else:
            save_model(model, output_dir, epoch)


def eval_model(model, eval_dataloader, device, training_type, distribute_type):
    with torch.no_grad():
        running_loss = 0.0
        eval_len = 1
        for index, data in enumerate(eval_dataloader):
            inputs = data['inputs'].to(device)
            inputs_mask = data["inputs_mask"].to(device)
            item_ids = data["item_ids"].to(device)

            if training_type == "mixed_precision":
                with autocast():
                    pooled_logits = model(inputs, inputs_mask, item_ids)
                    if distribute_type == "ddp":
                        loss = model.module.unsup_loss(pooled_logits)
                    else:
                        loss = model.unsup_loss(pooled_logits)
            else:
                pooled_logits = model(inputs, inputs_mask, item_ids)
                if distribute_type == "ddp":
                    loss = model.module.unsup_loss(pooled_logits)
                else:
                    loss = model.unsup_loss(pooled_logits)
            running_loss += loss.item()
            eval_len = index + 1
        print(f'Validation loss: {running_loss / eval_len}')
        return running_loss / eval_len


def save_model(model, output_dir, epoch):
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sub_path = os.path.join(output_dir, f"epoch_{str(epoch)}")
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    model.model.save_pretrained(sub_path)
    model_path = os.path.join(sub_path, "adapter.pth")
    item_embeddings, input_proj, output_proj = model.item_embeddings.state_dict(), \
        model.input_proj.state_dict(), \
        model.output_proj.state_dict()
    torch.save({'item_embeddings': item_embeddings, 'input_proj': input_proj, 'output_proj': output_proj}, model_path)


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

    random_seed = 2024
    # to enable more deterministic results.
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    print("random seed:", random_seed)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)
    torch.cuda.empty_cache()

    mtp_train_config_file = "/opt/huawei/schedule-train/algorithm/train.config"
    if os.path.exists(mtp_train_config_file):
        args.train_config_file = mtp_train_config_file

    start_time = time.time()
    args.num_gpus = torch.cuda.device_count()
    print("torch.cuda.is_available(): ", torch.cuda.is_available())
    print("torch.cuda.device_count(): ", args.num_gpus)

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
