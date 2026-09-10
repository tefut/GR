import collections
import argparse
import json
import os
import pickle
import time
import random
import stat

import numpy as np
import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from datasets.metaid_dataset import MetaIDDataset
from models.metaid_ae_v1 import MetaIDAEV1
from data_process import metaid_v1_data_process


def train(local_rank, args):
    data_path = args.data_path
    output_dir = args.output_dir
    keep_embd_dir = args.keep_embd_dir
    distribute_type = args.distribute_type
    train_config_file = args.train_config_file

    with open(train_config_file, 'r', encoding='utf-8') as fin:
        train_config = json.load(fin)
    print(train_config)

    distribute_type = distribute_type.lower()
    print("distribute_type:", distribute_type)

    input_df, feats_dict = metaid_v1_data_process(
        data_path, train_config.get('sel_cols', ['song_id']), 
        train_config.get('all_col_nm', ['song_id']), 
        train_config.get('file_type', 'orc')
        )

    # 构造模型
    exclude_ids = train_config.get("exclude_ids", ['theme_tag_id'])
    id_dim = train_config.get("id_dim", 8)
    embd_dim = train_config.get("embd_dim", 8)
    with_heads = train_config.get("with_heads", 1)
    with_heads = True if with_heads == 1 else False
    head_layers = train_config.get("head_layers", 2)

    model = MetaIDAEV1(feats_dict, exclude_ids, id_dim, embd_dim, with_heads, head_layers)

    batch_size = train_config.get("batch_size", 16)
    if distribute_type == "dp":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training_dataset = MetaIDDataset(input_df, feats_dict, device)
        model = DataParallel(model)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    elif distribute_type == "ddp":
        torch.cuda.set_device(local_rank)
        args.rank = local_rank
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.num_gpus, rank=args.rank)
        device = torch.device("cuda", local_rank)
        training_dataset = MetaIDDataset(input_df, feats_dict, device)
        model.to(device)
        model = DDP(model, device_ids=[local_rank])

        train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=True)

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training_dataset = MetaIDDataset(input_df, feats_dict, device)
        model.to(device)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    opt = Adam(model.parameters(), lr=train_config.get("init_learning_rate", 1e-2))

    # 开始训练
    print("start training ...")
    num_epoch = train_config.get("num_epochs", 90)
    lr_steps = train_config.get("lr_steps", [70])
    save_interval = train_config.get("save_interval", 10)
    schd = MultiStepLR(opt, lr_steps, gamma=0.1)
    lossFn = nn.MSELoss()
    recon_loss_weight = train_config.get("recon_loss_weight", 1.0)
    aux_cls_lossFn = nn.CrossEntropyLoss()
    aux_weight = dict()
    for k, v in zip(train_config.get("aux_loss_key", 'key'), train_config.get("aux_loss_weight", 0.1)):
        aux_weight[k] = float(v)
    is_fix_embd = train_config.get("is_fix_embd", 0)
    is_fix_embd = True if is_fix_embd == 1 else False

    best_loss = float('inf')
    best_epoch = 0

    if is_fix_embd:
        for param in model.embd_layer.embd_dict.parameters():
            param.requires_grad = False

    for epoch in range(num_epoch):
        model.train()
        
        running_loss = 0.0
        step_cnter = 0
        aux_losses_dict = collections.defaultdict(float)

        if distribute_type == "ddp":
            training_dataloader.sampler.set_epoch(epoch)
        for train_in in training_dataloader:

            raw, recon, _, logits_lst, keys_lst = model(train_in)

            loss = recon_loss_weight * lossFn(recon, raw)
            for k, logit in zip(keys_lst, logits_lst):
                curr_aux_weight = aux_weight.get(k, 0.1)
                cls_loss = curr_aux_weight * aux_cls_lossFn(logit, train_in[k])
                loss += cls_loss
                aux_losses_dict[k] += cls_loss.item()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)  # 梯度裁剪
            opt.step()
            schd.step()

            running_loss += loss.item()
            step_cnter += 1

        print('====> Epoch: {} Average loss: {:.9f}'.format(epoch, running_loss / step_cnter))
        for loss_k, loss_v in aux_losses_dict.items():
            print('====> AUX: {} Average loss: {:.9f}'.format(loss_k, loss_v / step_cnter))

        if running_loss / step_cnter > best_loss:
            continue
        best_loss = running_loss / step_cnter
        # 保存模型
        if epoch % save_interval == 0:
            if distribute_type.endswith("dp"):
                if dist.get_rank() == 0:
                    save_model(model.module, output_dir, epoch)
            else:
                save_model(model, output_dir, epoch)
            best_epoch = epoch


    print("write data ...")
    model.load_state_dict(
        torch.load(os.path.join(output_dir, f"epoch_{str(best_epoch)}", "ae_metaid_v1.pth"), map_location=device)[
            'state_dict'
        ]
    )
    model.eval()

    embd_lst, encode_song_id_lst = list(), list()
    for all_inputs in training_dataset:
        encode_song_id = all_inputs['song_id'].item()
        _, _, encode_embd, _, _ = model(all_inputs)
        embd_lst.append(encode_embd.detach().cpu())
        encode_song_id_lst.append(encode_song_id)
    embd_lst = torch.stack(embd_lst, dim=0).numpy()
    encode_song_id_lst = np.array(encode_song_id_lst)

    rev_feat_song_id_dict = dict()
    for k, v in feats_dict['song_id'].items():
        rev_feat_song_id_dict[v] = k

    if not os.path.exists(keep_embd_dir):
        os.makedirs(keep_embd_dir)
    item_embeddings_output = write_to_file(os.path.join(keep_embd_dir, 'item_embd.csv'), 'w')
    for encode_song_id, item_embed in zip(encode_song_id_lst, embd_lst):
        item_embed = item_embed / np.linalg.norm(item_embed)
        song_id = rev_feat_song_id_dict[encode_song_id]
        if np.any(np.isnan(item_embed)) or np.all(item_embed == 0):
            print(f"error embedding: {item_embed}")
            continue
        emb = ",".join(map(str, item_embed.tolist()))
        item_embeddings_output.write(str(song_id) + "|" + str(emb) + "\n")


def write_to_file(save_file, mode='w'):
    flags = os.O_WRONLY | os.O_CREAT
    stats = stat.S_IWUSR | stat.S_IRUSR
    file_hander = os.fdopen(os.open(save_file, flags, stats), mode)
    return file_hander


def save_model(model, output_dir, epoch):
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sub_path = os.path.join(output_dir, f"epoch_{str(epoch)}")
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    model_path = os.path.join(sub_path, "ae_metaid_v1.pth")
    state = dict()
    state['state_dict'] = model.state_dict()
    
    torch.save(state, model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', required=True, type=str, help='Please specify the path of dataset')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='Please specify the path of output')
    parser.add_argument(
        '-i', '--keep_embd_dir', required=True, type=str, help='Please specify the path of item embedding file'
    )
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
