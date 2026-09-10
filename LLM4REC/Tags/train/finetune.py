import argparse
import os.path
import random
import numpy as np
import time
import json
import pickle

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW
from dataset import IMCIDataset, IMCIECollator
from torch.utils.data import DataLoader
from model import BaseModel, IMCIEModel
from utils import load_file, calc_loss_batch, evaluate_model


def run(local_rank, args):
    base_model = args.base_model
    data_path = args.data_path
    cache_dir = args.cache_dir
    output_dir = args.output_dir
    train_config_file = args.train_config_file
    distribute_type = args.distribute_type
    item_embedding_file = args.item_embedding_file

    with open(train_config_file, 'r', encoding='utf-8') as fin:
        train_config = json.load(fin)
    print(train_config)

    max_length = train_config.get("max_length", 128)
    train_rate = train_config.get("train_rate", 0.8)
    test_rate = train_config.get("test_rate", 0.1)
    data_file = os.path.join(data_path, "game_info.json")
    data = load_file(data_file)
    embedding_mode = train_config.get("embedding_mode", "mean_pooling")

    item_embeds = pickle.load(open(item_embedding_file, 'rb'))

    train_portion = int(len(data) * train_rate)
    test_portion = int(len(data) * test_rate)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    eval_data = data[train_portion + test_portion:]

    train_dataset = IMCIDataset(train_data)
    test_dataset = IMCIDataset(test_data)
    eval_dataset = IMCIDataset(eval_data)

    data_collator = None
    ignore_index, device = -100, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_collator = IMCIECollator(base_model, cache_dir, max_length, ignore_index, device)

    input_dim = train_config.get("input_dim", 64)
    output_dim = train_config.get("output_dim", 32)

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
        embedding_mode=embedding_mode
    )

    print(gen_model)
    print(cf_model)

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
        gen_model = DataParallel(gen_model)
        cf_model = DataParallel(cf_model)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True)

    elif distribute_type == "ddp":
        torch.cuda.set_device(local_rank)
        args.rank = local_rank
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.num_gpus, rank=args.rank)
        device = torch.device("cuda", local_rank)
        gen_model.to(device)
        cf_model.to(device)
        gen_model = DDP(gen_model, device_ids=[local_rank])
        cf_model = DDP(cf_model, device_ids=[local_rank])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, \
                                      sampler=train_sampler)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen_model.to(device)
        cf_model.to(device)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

    # 是否进行混合精度训练
    if training_type == "mixed_precision":
        scaler = GradScaler()
    else:
        gen_model.float()
        cf_model.float()

    train(gen_model, cf_model, train_config, training_type, distribute_type, train_dataloader, eval_dataloader, \
          output_dir, device)


def train(gen_model, cf_model, train_config, training_type, distribute_type, train_dataloader, eval_dataloader,
          output_dir, device):
    if training_type == "mixed_precision":
        scaler = GradScaler()
    else:
        gen_model.float()
        cf_model.float()

    # 设置可训练参数，并构造优化器
    for name, param in gen_model.named_parameters():
        if "lora" in name or "input_proj" in name or "output_proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    gen_parameters = filter(lambda p: p.requires_grad, gen_model.parameters())
    for name, param in cf_model.named_parameters():
        if "lora" in name or "input_proj" in name or "output_proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    cf_parameters = filter(lambda p: p.requires_grad, cf_model.parameters())
    optimizer = AdamW(list(gen_parameters) + list(cf_parameters),
                      lr=train_config.get("learning_rate", 0.00003),
                      betas=(0.8, 0.999), weight_decay=3e-7)

    print("start training ...")
    num_epoch = train_config.get("num_epochs", 1)
    best_val_loss = float('inf')
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epoch):
        running_loss = 0.0
        if distribute_type == "ddp":
            train_dataloader.sampler.set_epoch(epoch)

        for collaborative_info, content_info in train_dataloader:
            index, data = collaborative_info
            input_batch, target_batch = content_info
            inputs = data['inputs'].to(device)
            inputs_mask = data["inputs_mask"].to(device)
            item_ids = data["item_ids"].to(device)

            if training_type == "mixed_precision":
                with autocast():
                    gen_output = gen_model(input_batch)
                    cf_logits = cf_model(inputs, inputs_mask, item_ids)
                    gen_logits = gen_output.logits.flatten(0, 1)
                    if distribute_type == "ddp":
                        cf_loss = cf_model.module.unsup_loss(cf_logits)
                    else:
                        cf_loss = cf_model.unsup_loss(cf_logits)
                    gen_loss = torch.nn.functional.cross_entropy(gen_logits, target_batch.flatten())
                loss = cf_loss + gen_loss
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(gen_parameters, max_norm=20)
                torch.nn.utils.clip_grad_norm_(cf_parameters, max_norm=20)
                scaler.step(optimizer)
                scaler.update()
            else:
                gen_output = gen_model(input_batch)
                cf_logits = cf_model(inputs, inputs_mask, item_ids)
                gen_logits = gen_output.logits.flatten(0, 1)
                if distribute_type == "ddp":
                    cf_loss = cf_model.module.unsup_loss(cf_logits)
                else:
                    cf_loss = cf_model.unsup_loss(cf_logits)
                    gen_loss = torch.nn.functional.cross_entropy(gen_logits, target_batch.flatten())
                loss = cf_loss + gen_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gen_parameters, max_norm=20)
                torch.nn.utils.clip_grad_norm_(cf_parameters, max_norm=20)
                optimizer.step()
            running_loss += loss.item()
            log_interval = train_config.get("log_interval", 1000)
            if index % log_interval == (log_interval - 1):
                print(f'[Epoch:{epoch + 1} Step:{index + 1} loss: {running_loss / log_interval}]')
                running_loss = 0.0
        val_loss = eval_model(gen_model, cf_model, eval_dataloader, device, training_type, distribute_type)
        if val_loss > best_val_loss:
            continue
        best_val_loss = val_loss

        print("start save model ...")
        if distribute_type.endswith("dp"):
            if dist.get_rank() == 0:
                save_model(gen_model.module, output_dir + '/gen_model.pth', epoch)
                save_model(cf_model.module, output_dir + '/cf_model.pth', epoch)
        else:
            save_model(gen_model, output_dir + '/gen_model.pth', epoch)
            save_model(cf_model, output_dir + '/cf_model.pth', epoch)


def eval_model(gen_model, cf_model, eval_dataloader, device, training_type, distribute_type):
    with torch.no_grad():
        running_loss = 0.0
        eval_len = 1
        for collaborative_info, content_info in eval_dataloader:
            index, data = collaborative_info
            input_batch, target_batch = content_info
            inputs = data['inputs'].to(device)
            inputs_mask = data["inputs_mask"].to(device)
            item_ids = data["item_ids"].to(device)

            if training_type == "mixed_precision":
                with autocast():
                    gen_output = gen_model(input_batch)
                    cf_logits = cf_model(inputs, inputs_mask, item_ids)
                    gen_logits = gen_output.logits.flatten(0, 1)
                    if distribute_type == "ddp":
                        cf_loss = cf_model.module.unsup_loss(cf_logits)
                    else:
                        cf_loss = cf_model.unsup_loss(cf_logits)
                    gen_loss = torch.nn.functional.cross_entropy(gen_logits, target_batch.flatten())
            else:
                gen_output = gen_model(input_batch)
                cf_logits = cf_model(inputs, inputs_mask, item_ids)
                gen_logits = gen_output.logits.flatten(0, 1)
                if distribute_type == "ddp":
                    cf_loss = cf_model.module.unsup_loss(cf_logits)
                else:
                    cf_loss = cf_model.unsup_loss(cf_logits)
                gen_loss = torch.nn.functional.cross_entropy(gen_logits, target_batch.flatten())
            loss = cf_loss + gen_loss
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
    input_proj, output_proj = model.input_proj.state_dict(), model.output_proj.state_dict()
    torch.save({'input_proj': input_proj, 'output_proj': output_proj}, model_path)


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
    parser.add_argument('-t', '--train_config_file', required=True, type=str,
                        help='Please specify the path of train config')
    parser.add_argument('-r', '--distribute_type', default="general", type=str)

    random_seed = 2024
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    print("random seed:", random_seed)

    args, unknown = parser.parse_known_args()
    print("unknown arguments: ", unknown)
    print("arguments: ", args)
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
        run(local_rank=0, args=args)
    elif args.num_gpus > 1 and args.distribute_type.lower() == "ddp":
        torch.multiprocessing.set_start_method("spawn")
        port_id = 10000 + np.random.randint(0, 1000)
        args.dist_url = "tcp://127.0.0.1:" + str(port_id)
        print(args)
        mp.spawn(run, nprocs=args.num_gpus, args=(args,))
    else:
        local_rank = os.environ.get("LOCAL_RANK", 0)
        run(local_rank=local_rank, args=args)
    end_time = time.time()
    print(f"training time: {end_time - start_time}")


if __name__ == '__main__':
    main()
