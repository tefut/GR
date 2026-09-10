import json
import time
import os
import random
import argparse
import numpy as np
import pandas as pd
import logging
import torch

try:
    import torch_npu
except Exception as e:
    logging.info("torch_npu not installed: %s", e)

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from recall.recall_metric import Metrics
from recall.recall_model import SeqModel
from recall.recall_dataset import GRDataset, collate_fn


# 分布式环境初始化
def setup(device_type):
    if device_type == "cpu":
        device = "cpu"
        rank_id = 0
        world_size = 1
        local_rank = 0
        return device, rank_id, local_rank, world_size

    addr = os.getenv("MASTER_ADDR")
    port = os.getenv("MASTER_PORT")
    rank_id = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    tcp_url = f"tcp://{addr}:{port}/rank={rank_id}"
    print("tcp url %s", tcp_url)

    if device_type == "gpu":
        num_gpus = torch.cuda.device_count()
        print("torch.cuda.is_available(): ", torch.cuda.is_available())
        print("torch.cuda.device_count(): ", num_gpus)
        dist.init_process_group("nccl", init_method=f"tcp://{addr}:{port}", rank=rank_id,
                                world_size=world_size)
        device = torch.device("cuda", local_rank)
    elif device_type == "npu":
        num_npus = torch.npu.device_count()
        print("torch.npu.is_available(): ", torch.npu.is_available())
        print("torch.npu.device_count(): ", num_npus)
        dist.init_process_group("hccl", init_method=f"tcp://{addr}:{port}", rank=rank_id, world_size=world_size)
        torch.npu.set_device(local_rank)
        device = torch.device("npu", local_rank)
    else:
        print("model run in cpu")
        device = "cpu"

    return device, rank_id, local_rank, world_size


def cleanup():
    dist.destroy_process_group()


def init_random_seed(random_seed):
    """
    设置全局随机数以使训练结果更确定
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def init_npu_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch_npu.npu.manual_seed(random_seed)
    torch_npu.npu.manual_seed_all(random_seed)


def load_dataset(dataset_args, local_rank, world_size):
    # 获取dataset和dataloader的相关参数
    train_dataset_config = dataset_args.get("train_dataset_config")
    val_dataset_config = dataset_args.get("valid_dataset_config")
    train_dataloader_config = dataset_args.get("train_dataloader_config")
    val_dataloader_config = dataset_args.get(
        "val_dataloader_config") if "val_dataloader_config" in dataset_args else train_dataloader_config

    # 构建dataset和dataloader
    train_dataset = GRDataset(
        **train_dataset_config
    )
    val_dataset = GRDataset(**val_dataset_config)

    # 设置数据分布处理
    train_dataset.set_distributed(world_size, local_rank)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        **train_dataloader_config
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        collate_fn=collate_fn,
        **val_dataloader_config
    )
    return (train_dataloader, val_dataloader)


def load_model(model_args, rank, device):
    model = SeqModel(model_args)
    if device == "cpu":
        return model
    model = model.to(device)
    model = DDP(model, device_ids=[rank], broadcast_buffers=False,
                find_unused_parameters=False)
    return model


def load_optimizer(optimizer_args, model):
    optimizer_type = optimizer_args.get("type")
    learning_rate = optimizer_args.get("learning_rate")
    betas = tuple(optimizer_args.get("beta"))
    weight_decay = optimizer_args.get("weight_decay")

    if optimizer_type.strip().lower() == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    elif optimizer_type.strip().lower() == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    elif (optimizer_type.strip().lower() == "sgd"):
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unrecognized optimizer '{optimizer_type}'..")
    return optim


def load_metric(metric_args):
    if metric_args is not None:
        metric = Metrics(**metric_args)
    else:
        metric = Metrics()
    return metric


def train(train_args, model, dataloader, optimizer, metric):
    device = train_args.get("device")
    rank = train_args.get("rank")
    num_epoch = train_args.get("num_epoch")
    eval_step = train_args.get("eval_step", 1000)
    clipnorm = train_args.get("clipnorm")
    save_model_path = train_args.get("save_model_path")
    total_loss = torch.tensor(0.0, device=device)

    train_dataloader, val_dataloader = dataloader

    for epoch in range(num_epoch):
        # 同步所有进程
        dist.barrier()
        for bid, batch in enumerate(train_dataloader):
            did = batch.pop("did")

            past_ids = batch.get("app_id_list").to(device)
            past_lengths = batch.get("app_id_list_size").to(device)
            target_ids = batch.get("target_app_id").to(device)

            past_lengths = torch.clamp(past_lengths.int(), max=past_ids.size(1))

            seq_embeddings = model(past_ids, past_lengths)

            if device == "cpu":
                loss = model.get_loss(past_ids, past_lengths, seq_embeddings)
            else:
                loss = model.module.get_loss(past_ids, past_lengths, seq_embeddings)
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=clipnorm)  # 梯度裁剪

            optimizer.step()
            total_loss += loss.detach()

            # 每eval_step个batch评估一次
            if bid % eval_step == (eval_step - 1):
                user_embeddings = model.module.get_user_embedding(past_lengths, seq_embeddings, curr=2)
                predict_logits = model.module.get_topK_logits(user_embeddings)
                score = metric(predict_logits, target_ids)
                print(f'[Process:{rank} Epoch:{epoch + 1} Step:{bid + 1} '
                      f'train_loss: {total_loss / eval_step} score: {json.dumps(score, indent=4)}]')
                total_loss = torch.tensor(0.0, device=device)
                metric.reset()
        # 同步所有进程
        dist.barrier()

        if rank == 0:
            print("evaluate model ... ")
            save_model(model, save_model_path)
            # 每个epoch结束后的评估
            val_score = eval_model(val_dataloader, model, metric, device, save_model_path)

            print(f"Epoch {epoch + 1} - Train Loss: {total_loss / (bid % eval_step):.4f}, "
                  f"Valid score:{json.dumps(val_score, indent=4)}")

    return total_loss


@torch.no_grad()
def eval_model(val_dataloader, model, metric, device, save_model_path):
    model.eval()
    metric.reset()
    user_embeddings_data = {"user_id": [], "user_embedding": []}
    for bid, batch in enumerate(val_dataloader):
        did = batch.pop("did")
        user_embeddings_data["user_id"].extend(did)

        past_ids = batch.get("app_id_list").to(device)
        past_lengths = batch.get("app_id_list_size").to(device)
        past_lengths = torch.clamp(past_lengths.int(), max=past_ids.size(1))

        target_ids = batch.get("target_app_id").to(device)

        seq_embeddings = model(past_ids, past_lengths)
        curr_embeddings = model.module.get_user_embedding(past_lengths, seq_embeddings, curr=2)
        predict_logits = model.module.get_topK_logits(curr_embeddings)
        score = metric(predict_logits, target_ids)

        user_embeddings = model.module.get_user_embedding(past_lengths, seq_embeddings)
        user_embeddings_list = user_embeddings.detach().cpu().numpy().tolist()
        user_embed_str = [",".join(map(str, embed)) for embed in user_embeddings_list]
        user_embeddings_data["user_embedding"].extend(user_embed_str)

        if bid % 500 == 499:
            val_score = metric.compute()
            print(f"Batch {bid} - score: {json.dumps(val_score, indent=4)}")
    score = metric.compute()
    user_embedding_file = os.path.join(save_model_path, "user_embeddings.csv")
    user_embeddings_df = pd.DataFrame(user_embeddings_data)
    user_embeddings_df.to_csv(user_embedding_file, header=False, index=False, sep="|", encoding='utf-8')

    item_embedding_file = os.path.join(save_model_path, "item_embeddings.csv")
    item_embeddings = model.module.get_item_embeddings(turn_id=True)
    item_embeddings_df = pd.DataFrame(item_embeddings)
    item_embeddings_df.to_csv(item_embedding_file, header=False, index=False, sep="|", encoding='utf-8')

    return score


def save_model(model, save_model_path):
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    torch.save(model.module.state_dict(), os.path.join(save_model_path, 'best_dcn_model.pth'))


def run_training(config_file, device_type):
    # 加载配置文件
    with open(config_file, 'r', encoding='utf-8') as fin:
        train_config = json.load(fin)
    print("train_config:", train_config)

    # 初始化分布训练环境
    device, rank, local_rank, world_size = setup(device_type)

    # 获取dataloader, model, optimizer, training的相关参数
    dataset_args = train_config.get("dataset_configs")
    model_args = train_config.get("model_configs")
    optimizer_args = train_config.get("optimizer_configs")
    metric_args = train_config.get("metric_configs", None)
    train_args = train_config.get("train_configs")
    train_args["device"] = device
    train_args["rank"] = rank
    train_args["world_size"] = world_size

    random_seed = train_args.get("global_seed", 2025)
    if device_type == "npu":
        init_npu_random_seed(random_seed)
    else:
        init_random_seed(random_seed)

    # 加载dataloader, model, optimizer
    dataloader = load_dataset(dataset_args, local_rank, world_size)
    model = load_model(model_args, rank, device)
    optimizer = load_optimizer(optimizer_args, model)
    metirc = load_metric(metric_args)

    # 开始训练
    train(train_args, model, dataloader, optimizer, metirc)

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device_type', required=True, type=str,
                        help='Please specify device type gpu or npu')

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)
    torch.cuda.empty_cache()

    mtp_train_config_file = "/opt/huawei/schedule-train/algorithm/train.config"
    if os.path.exists(mtp_train_config_file):
        args.config_file = mtp_train_config_file

    start_time = time.time()

    run_training(args.config_file, args.device_type)

    end_time = time.time()
    print(f"training time: {end_time - start_time}")


if __name__ == '__main__':
    main()
