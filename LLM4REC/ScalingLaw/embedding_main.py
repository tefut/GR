import copy
import json
import logging
import os
import pickle
import random
import sys
import time
from math import sqrt
from statistics import mean
from typing import Any, Callable, Dict, Tuple, Iterable, Union
import queue
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from absl import app, flags
from sklearn.metrics import roc_auc_score
from torch import optim
from modeling.model_registry import ModelRegistry
from modeling.model_initializer import ModelInitializer
from torch.nn.parallel import DistributedDataParallel as DDP
from data.data_loader import create_data_loader
from data.reco_dataset import get_reco_dataset

from modeling.generic.utils.constants import Const
from data.eval import gather_all_list, avg_eval
from utils.common_utils import get_config, load_feature_map, refine_feat_and_model_conf
from embedding.eval import eval_recall_metrics, get_eval_state
from embedding.indexing import get_top_k_module
from torch.utils.tensorboard import SummaryWriter

# 初始化传入参数
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

flags.DEFINE_string("config_file", None, "Path to the config file.")
flags.DEFINE_integer("master_port", 12355, "Master port.")
flags.DEFINE_string("data_dir", None, "Path to data.")
flags.DEFINE_string("save_dir", None, "Path to save.")
flags.DEFINE_string("emb_dir", None, "Path to save embedding.")
flags.DEFINE_string("feature_map_dir", None, "Path of feature_map.")
flags.DEFINE_string("feature_max_index_path", None, "Path of feature map max index")
flags.DEFINE_string("period", None, "Period of task execution.")
flags.DEFINE_string("tensorboard_log_dir", None, "Period of save log.")
flags.DEFINE_boolean("is_train", True, "If the model is training or testing.")
flags.DEFINE_boolean("save_user_emb", True, "If the model is used for saving user emb")

FLAGS = flags.FLAGS

train_data_queue = mp.Queue(maxsize=3)
eval_data_queue = mp.Queue(maxsize=3)
generate_emb_data_queue = mp.Queue(maxsize=3)


def init_ddp_info_params(already_init=False):
    """
    初始化加速器参数
    """

    addr = os.getenv("MASTER_ADDR")
    port = os.getenv("MASTER_PORT")
    rank_id = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_num = int(os.environ.get("NNODES", 1))
    logging.info(f"tcp://{addr}:{port}, nnodes={node_num}, rank={rank_id}")

    # initialize the process group
    if not already_init:
        # initialize the process group
        dist.init_process_group("nccl", init_method=f"tcp://{addr}:{port}", rank=rank_id,
                                world_size=world_size)

    return f"cuda:{local_rank}", rank_id, local_rank, world_size, node_num


def async_data_loader(data_loader, data_queue, is_train=True, num_epochs=1, batch_limit=0):
    # for _ in range(num_epochs):
    for idx, row in enumerate(iter(data_loader)):
        if is_train and batch_limit and idx > batch_limit:
            break
        data_queue.put(row)

    data_queue.put("end")
    # Sleep to ensure the result of `queue.empty()` absolutely accurate
    time.sleep(1e-3)
    while not data_queue.empty():
        time.sleep(1)
    return


def init_random_seed(config):
    """
    设置全局随机数以使训练结果更确定
    """
    random_seed = config.get('seed_conf', {}).get("global_seed", '1234')
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def init_torch_config(local_rank, train_conf):
    """
    设置torch配置
    """
    use_tf32 = train_conf.get("use_tf32", False)
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    logging.info("cuda.matmul.allow_tf32: %s", use_tf32)
    logging.info("cudnn.allow_tf32: %s", use_tf32)


def get_data_loaders(dataset, data_dir, rank, world_size, train_config, model_conf, feature_config, dataloader_config):
    """
    获取训练和验证数据 dataloader
    """
    dataset, eval_data_loader, train_data_loader = get_train_eval_dataloader(
        dataset, data_dir, rank, world_size, train_config, model_conf, feature_config, dataloader_config,
        is_recall=train_config['is_recall']
    )
    return dataset, eval_data_loader, train_data_loader


def get_optimizer(train_conf, model, learning_rate):
    """
    获取优化器和损失函数
    """
    beta = tuple(train_conf["beta"])
    optimizer_type = train_conf["optimizer_type"]
    weight_decay = train_conf["weight_decay"]
    opt_dict = {
        "adamw": torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=beta, weight_decay=weight_decay),
        "adam": torch.optim.Adam(model.parameters(), lr=learning_rate, betas=beta, weight_decay=weight_decay),
        "sgd": torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
    }
    if optimizer_type.strip().lower() not in opt_dict:
        raise ValueError("Unknown optimizer_type %s" % optimizer_type)
    return opt_dict[optimizer_type.strip().lower()]


def train_step(model, opt, train_conf,
               batch_id, world_size,
               rank, epoch, device):
    
    """
    训练模型
    """
    model.train()
    train_costs, batch_group = [], train_conf['eval_interval']
    last_training_time = time.time()
    is_output_loss_csv_file = train_conf.get("is_output_loss_csv_file", False)
    step_list, loss_list = [], []

    while True:
        try:
            model.train()
            seq_features_cpu = train_data_queue.get()
            if seq_features_cpu == "end":
                break
            payloads_tod = {}
            
            for name, feature in seq_features_cpu.items():
                payloads_tod[name] = feature.to(device, non_blocking=True)
            
            B = payloads_tod.get("uid").size(0)
            if B == 0:
                break
            opt.zero_grad()

            model_input = payloads_tod
            model_input['past_ids'] = model_input['sequence_item_ids']
            
            loss = model(model_input)
            loss.backward()
            opt.step()
            loss_list.append(loss.detach().cpu().item())
            step_list.append(batch_id)
            if (batch_id % batch_group) == 0:
                train_cost = time.time() - last_training_time
                logging.info("rank %s; batch-stat (train): step %s "
                             "(epoch %s in %.2fs): %.6f", rank, batch_id, epoch, train_cost,
                             mean(loss_list[-batch_group:]))
                train_costs.append(train_cost)
                last_training_time = time.time()
            batch_id += 1
        except queue.Empty:
            logging.info("Queue empty, wait for data loading...")
            
    if train_costs:
        logging.info(f"rank {rank} epoch {epoch}; mean cost per step: {mean(train_costs) / batch_group:.2f}s")
    else:
        logging.info(f"rank {rank} epoch {epoch}; less than {batch_group} steps")

    if is_output_loss_csv_file:
        loss_total_list = gather_all_list(loss_list, world_size=world_size)
        save_loss_csv_file(loss_total_list, step_list)

    epoch_loss = avg_eval(torch.tensor([loss]).to(device), world_size=world_size)

    return batch_id, epoch_loss


def save_loss_csv_file(loss_list, step_list):
    """
    保存loss.csv文件
    """
    save_dir = FLAGS.save_dir
    if not loss_list:  # 处理空列表的情况
        logging.info("loss_list is None!!!")
        pass

    step_total = len(step_list)  # 101 m/n
    all_loss_num = len(loss_list)  # 404 n
    step_length = all_loss_num / step_total  # 4 m
    # 分割成m组，每组长度为k
    sublists = [loss_list[i * step_total: (i + 1) * step_total] for i in range(int(step_length))]
    # 转置后计算每列的平均值
    score_avg_list = [sum(values) / step_length for values in zip(*sublists)]
    setp_avg_dict = {k: v for k, v in zip(step_list, score_avg_list)}

    df = pd.DataFrame({
        'step': list(setp_avg_dict.keys()),
        'loss': [round(v, 6) for v in setp_avg_dict.values()]
    })
    # 按键排序
    df = df.sort_values('step')
    # 保存为 CSV
    save_loss_path = "%s/modelfile" % save_dir
    loss_file_name = "step_vs_loss.csv"
    loss_file_path = os.path.join(save_loss_path, loss_file_name)
    if not os.path.exists(save_loss_path):
        os.makedirs(save_loss_path, exist_ok=True)
    logging.info("Saving step_vs_loss.csv to %s", loss_file_path)
    df.to_csv(loss_file_path, index=False)


def save_model_state_dict(model, save_dir, export_conf):
    model_file_name = "model_hstu.pth"
    export_dir = os.path.join(save_dir, export_conf["save_dir_name"])
    model_path = os.path.join(export_dir, model_file_name)

    # 路径不存在则创建
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
        logging.info("Created directory: %s", export_dir)

    model.eval()
    logging.info("Saving model to %s", model_path)
    torch.save(model.state_dict(), model_path)


def save_model(model,
               save_dir,
               export_conf):
    if not os.path.exists(os.path.join(save_dir, export_conf["save_dir_name"])):
        os.makedirs(os.path.join(save_dir, export_conf["save_dir_name"]), exist_ok=True)

    save_model_state_dict(model, save_dir, export_conf)


def evaluate_step(model, feature_conf, train_conf, world_size, rank, 
                  device, all_item_ids, filter_invalid_ids=False, epoch=0, writer=None):
    """
    评估模型
    """
    model.eval()
    logging.info("rank %s starting evaluation...", rank)

    save_score_csv_item_cols = feature_conf.get('save_score_csv_item_cols', {})
    save_score_csv_user_cols = feature_conf.get('save_score_csv_user_cols', {})

    group_keys_all = {
        'uid': []
    }
    for key in save_score_csv_item_cols:
        group_keys_all[key] = []
    for key in save_score_csv_user_cols:
        group_keys_all[key] = []

    eval_iter = 0
    last_eval_time = time.perf_counter()

    top_k_method = "MIPSBruteForceTopK"
    eval_dict_all = None
    eval_state = get_eval_state(
        all_item_ids=all_item_ids,
        negatives_sampler=model.module.negative_sampler,
        top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
            top_k_method=top_k_method,
            model=model,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        ),
        device=model.device,
        float_dtype=None,
    )

    while True:
        try:
            eval_seq_features_cpu = eval_data_queue.get()
            if eval_seq_features_cpu == "end":
                break
            payloads_tod = {}
            for name, feature in eval_seq_features_cpu.items():
                payloads_tod[name] = feature.to(device, non_blocking=True)
            
            model_input = payloads_tod
            B, N = model_input.get("sequence_item_ids").size()
            if B == 0:
                break
            
            flattened_offsets = (
                    (model_input['past_lengths'] - 1)
                    + torch.arange(start=0, end=B, step=1, dtype=model_input['past_lengths'].dtype,
                                   device=model_input['past_lengths'].device) * N
            )
            target_ids = model_input['sequence_item_ids'].view(-1)[flattened_offsets].reshape(B, 1)
            pid = model_input['sequence_item_ids'].view(-1)
            pid[flattened_offsets] = 0
            model_input['sequence_item_ids'] = pid.reshape(B, -1)
            model_input['past_lengths'] = model_input['past_lengths'] - 1

            eval_dict = eval_recall_metrics(
                eval_state, model.module, model_input, target_ids,
                user_max_batch_size=train_conf["eval_batch_size"],
                dtype=None,
                filter_invalid_ids=filter_invalid_ids,
            )
            if eval_dict_all is None:
                eval_dict_all = {}
                for k, _ in eval_dict.items():
                    eval_dict_all[k] = []

            if (eval_iter % train_conf["eval_interval"]) == 0:
                torch.distributed.barrier()  # sync time taken
                cost = time.perf_counter() - last_eval_time
                logging.info(
                    f"rank {rank}; batch-stat (eval): step {eval_iter} (EVAL in {cost:.2f}s)")
                last_eval_time = time.perf_counter()

            for k, v in eval_dict.items():
                eval_dict_all[k] = eval_dict_all[k] + [v]
            del eval_dict
            eval_iter += 1
        except queue.Empty:
            logging.info("Queue empty, wait for data loading...")
        
    for k, v in eval_dict_all.items():
        eval_dict_all[k] = torch.cat(v, dim=-1)

    ndcg_10 = avg_eval(eval_dict_all.get("ndcg@10"), world_size=world_size)
    ndcg_50 = avg_eval(eval_dict_all.get("ndcg@50"), world_size=world_size)
    ndcg_100 = avg_eval(eval_dict_all.get("ndcg@100"), world_size=world_size)
    ndcg_200 = avg_eval(eval_dict_all.get("ndcg@200"), world_size=world_size)
    hr_10 = avg_eval(eval_dict_all.get("hr@10"), world_size=world_size)
    hr_50 = avg_eval(eval_dict_all.get("hr@50"), world_size=world_size)
    hr_100 = avg_eval(eval_dict_all.get("hr@100"), world_size=world_size)
    hr_200 = avg_eval(eval_dict_all.get("hr@200"), world_size=world_size)
    hr_500 = avg_eval(eval_dict_all.get("hr@500"), world_size=world_size)
    hr_1000 = avg_eval(eval_dict_all.get("hr@1000"), world_size=world_size)
    hr_2000 = avg_eval(eval_dict_all.get("hr@2000"), world_size=world_size)
    mrr = avg_eval(eval_dict_all.get("mrr"), world_size=world_size)

    if rank == 0:
        logging.info(f"Evaluation :\n"
                        f"rank {rank}: recall: "
                        f"NDCG@10 {ndcg_10:.4f}, NDCG@50 {ndcg_50:.4f}, NDCG@100 {ndcg_100:.4f}, "
                        f"NDCG@200 {ndcg_200:.4f}, "
                        f"HR@10 {hr_10:.4f}, HR@50 {hr_50:.4f}, HR@100 {hr_100:.4f}, HR@200 {hr_200:.4f}, "
                        f"HR@500 {hr_500:.4f}, HR@1000 {hr_1000:.4f}, HR@2000 {hr_2000:.4f}, "
                        f"MRR {mrr:.4f}")
        if writer:
            writer.add_scalar("eval_epoch/ndcg@10", ndcg_10, epoch)
            writer.add_scalar("eval_epoch/ndcg@50", ndcg_50, epoch)
            writer.add_scalar("eval_epoch/ndcg@100", ndcg_100, epoch)
            writer.add_scalar("eval_epoch/ndcg@200", ndcg_200, epoch)
            writer.add_scalar("eval_epoch/hr@10", hr_10, epoch)
            writer.add_scalar("eval_epoch/hr@50", hr_50, epoch)
            writer.add_scalar("eval_epoch/hr@100", hr_100, epoch)
            writer.add_scalar("eval_epoch/hr@200", hr_200, epoch)
            writer.add_scalar("eval_epoch/hr@500", hr_500, epoch)
            writer.add_scalar("eval_epoch/hr@1000", hr_1000, epoch)
            writer.add_scalar("eval_epoch/hr@2000", hr_2000, epoch)
            writer.add_scalar("eval_epoch/mrr", mrr, epoch)

    return


def save_item_embs(model, feature_name, feature_map_path, emb_dir, rank, device, emb_field_delimiter, emb_delimiter,
                   feature_map_field_name):
    """
    保存某个特征如 App.Id的item embedding
    每一行为 original_value, embedding
    """
    model.eval()
    logging.info(f"Generating item embeddings for feature: {feature_name}")
    
    if os.path.isdir(feature_map_path):
        _feature_map_path = os.join(feature_map_path, os.listdir(feature_map_path)[0])
        feature_map_path = _feature_map_path
        
    feature_id_to_original = load_feature_map(
        txt_path=feature_map_path,
        feature_map_field_name=feature_map_field_name
    )

    emb_layer = model.module.embedding_module._item_info_embs[feature_name]
    all_embeddings: torch.Tensor = emb_layer.weight.detach().to(device).cpu()
    total_items = all_embeddings.size(0)

    group_keys_all = {
        'original_value': [],
        'embedding': []
    }

    for fid in range(total_items):
        if fid == 0 or fid not in feature_id_to_original:
            continue
        original_value = feature_id_to_original[fid]
        emb = all_embeddings[fid].tolist()
        emb = emb_delimiter.join([str(d) for d in emb])
        group_keys_all['original_value'].append(original_value)
        group_keys_all['embedding'].append(emb)

    save_path = os.path.join(emb_dir, "item_embeddings")
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f"item_embedding_{feature_name}.csv")
    logging.info(f"Saving item embeddings to {output_file}")
    if rank == 0:
        with open(output_file, "w") as f:
            for a, b in zip(group_keys_all['original_value'], group_keys_all['embedding']):
                f.write(f"{a}{emb_field_delimiter}{b}\n")
    logging.info(f"Done saving item embedding to {output_file}")


def save_user_embs(model, train_conf, feature_map_path, emb_dir, rank, device,
                   emb_field_delimiter, emb_delimiter, feature_map_field_name):
    """
    评估模型
    """
    model.eval()
    logging.info(f"Generating user embeddings for rank ... {rank}")

    group_keys_all = {
        'uid': [],
        'embeddings': []
    }

    save_iter, part_iter = 0, 0
    last_eval_time = time.perf_counter()
    
    user_id_to_original = load_feature_map(
        txt_path=feature_map_path,
        feature_map_field_name=feature_map_field_name
    )
    
    def write_embs(rank, partition, group_keys, emb_dir):
        save_path = f"{emb_dir}/user_embeddings/"
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        user_embedding_file_name = f"user_embedding_part_{rank}_{partition}.csv"
        embedding_path_name = os.path.join(save_path, user_embedding_file_name)
        logging.info("Saving user embeddings to %s", embedding_path_name)
        user_embedding_df = pd.DataFrame({
            **group_keys
        })
        user_embedding_df.to_csv(embedding_path_name, header=False, index=False, sep=emb_field_delimiter)
        logging.info("user embeddings count %s", user_embedding_df['uid'].count())

    while True:
        try:
            seq_features = generate_emb_data_queue.get()
            if seq_features == "end":
                break
            payloads_tod = {}
            for name, feature in seq_features.items():
                payloads_tod[name] = feature.to(device, non_blocking=True)
            
            model_input = payloads_tod
            B, N = model_input.get("sequence_item_ids").size()
            if B == 0:
                break
            
            user_embeddings = model.module.encode(
                past_lengths=model_input['past_lengths'],
                model_inputs=model_input,
            )

            group_keys_all['uid'] += [user_id_to_original[uid] for uid in seq_features['uid'].reshape(-1).tolist()]
            user_embeddings_cpu = user_embeddings.detach().cpu().tolist()
            user_embeddings_cpu = [emb_delimiter.join([str(d) for d in t]) for t in user_embeddings_cpu]
            group_keys_all['embeddings'] += user_embeddings_cpu
            if (save_iter % train_conf["eval_interval"]) == 0:
                cost = time.perf_counter() - last_eval_time
                logging.info(
                    f"rank {rank}; batch-stat (SAVE USER EMBS): step {save_iter} (SAVE USER EMBS in {cost:.2f}s)")
                last_eval_time = time.perf_counter()
                # save user embs every eval_interval * 10 steps 
                if save_iter % (10 * train_conf["eval_interval"]) == 0 and save_iter != 0:
                    logging.info(f'dumping user file rank={rank} num_partition={part_iter}')
                    write_embs(rank, part_iter, group_keys_all, emb_dir)
                    group_keys_all = {
                        'uid': [],
                        'embeddings': []
                    }
                    part_iter += 1
                torch.distributed.barrier()  # sync time taken

            del user_embeddings
            del model_input
            save_iter += 1
        except queue.Empty:
            logging.info("Queue empty, wait for data loading...")

    write_embs(rank, part_iter, group_keys_all, emb_dir)


def train_fn(config, data_dir, save_dir, feature_map_dir_or_path, period) -> None:
    """
    训练函数

    :param config: 配置文件字典
    :param data_dir: 数据集路径
    :param save_dir: 模型保存路径
    :return:
    """
    # 1. 初始化随机数、加速器，配置文件
    # 1.1 这一部分可以通用
    common_config = config[Const.COMMON_HP]
    init_random_seed(common_config)
    device, rank, local_rank, world_size, node_num = init_ddp_info_params()
    if rank == 0 and FLAGS.tensorboard_log_dir:
        writer = SummaryWriter(log_dir=FLAGS.tensorboard_log_dir)
        logging.info(f"Rank {rank}: writing logs to {FLAGS.tensorboard_log_dir}")
    else:
        writer = None

    train_conf = common_config['train_conf']
    feature_conf = common_config['feature_conf']
    learning_rate = train_conf['learning_rate']
    lr_scaling = train_conf['lr_scaling']
    export_conf = common_config['export_conf']
    data_loader_conf = common_config['data_loader_conf']
    model_conf = common_config['model_conf']
    dataset_name = data_loader_conf["dataset_name"]
    feature_conf["period"] = period
    train_conf['learning_rate'] = init_learning_rate(learning_rate, lr_scaling, world_size)
    filter_invalid_ids = train_conf.get('filter_invalid_ids', False)

    logging.info("Training model on rank %s (local rank: %s); device: %s; world_size: %s;",
                 rank, local_rank, device, world_size)

    init_torch_config(local_rank, train_conf)

    feature_conf, model_conf, feature_map = refine_feat_and_model_conf(dataset=dataset_name,
                                                                       feature_conf=feature_conf,
                                                                       model_conf=model_conf,
                                                                       feature_map_dir_or_path=feature_map_dir_or_path,
                                                                       model_cfg=config[Const.MODEL_CFG])
    
    # 2. 初始化数据集
    dataset, eval_data_loader, train_data_loader = get_data_loaders(dataset=dataset_name,
                                                                    data_dir=data_dir,
                                                                    rank=rank,
                                                                    world_size=world_size,
                                                                    train_config=train_conf,
                                                                    model_conf=model_conf,
                                                                    feature_config=feature_conf,
                                                                    dataloader_config=data_loader_conf)
    # 3. 初始化模型
    ModelRegistry.register_all_modules(
        modul_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "modeling/generic/sequential")))
    model = ModelInitializer.init(gr_module_cfg=config)

    if hasattr(model, 'negative_sampler') and hasattr(model.negative_sampler, 'set_all_item_ids') and train_conf.get(
            'filter_invalid_ids', False):
        model.negative_sampler.set_all_item_ids(dataset.all_item_ids)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Total number of parameters: {total_params:.2f}M")
    total_params = sum(p.numel() for x, p in model.named_parameters() if '_emb' not in x) / 1e6
    logging.info(f"The number of parameters (exl. emb): {total_params:.2f}M")
    
    opt = get_optimizer(train_conf=train_conf,
                        model=model, learning_rate=learning_rate)
    
    # ------------------------------async process of data loader-----------------------------
    torch.set_num_threads(world_size)
    train_async_loader = mp.Process(target=async_data_loader,
                                    args=(train_data_loader, train_data_queue, True, train_conf['num_epochs']))
    train_async_loader.start()

    batch_id = 0 

    model.to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    
    for epoch in range(train_conf['num_epochs']):
        train_step_para = {
            "model": model,
            "opt": opt,
            "train_conf": train_conf,
            "batch_id": batch_id,
            "world_size": world_size,
            "rank": rank,
            "epoch": epoch,
            "device": device
        }

        batch_id, epoch_loss = train_step(**train_step_para)

        if rank == 0 and writer:
            logging.info("loss at epoch %s is %s", epoch, epoch_loss)
            writer.add_scalar("loss/train", epoch_loss, epoch)

        logging.info("saving model state dict")
        save_model_state_dict(model=model.module, save_dir=save_dir, export_conf=export_conf)

        if train_conf.get('eval_after_train', False):
            eval_async_loader = mp.Process(target=async_data_loader,
                                args=(eval_data_loader, eval_data_queue, True, 1))
            eval_async_loader.start()
            evaluate_step(model=model,
                          feature_conf=feature_conf,
                          train_conf=train_conf,
                          world_size=world_size,
                          rank=rank,
                          device=device,
                          all_item_ids=dataset.all_item_ids,
                          epoch=epoch,
                          filter_invalid_ids=filter_invalid_ids,
                          writer=writer)

    if rank == 0 and writer:
        writer.close()


def eval_fn(config, data_dir, save_dir, feature_map_dir_or_path, period, already_init=False) -> None:
    """
    单独评估函数
    """
    # 1. 初始化随机数、加速器，配置文件
    # 1.1 这一部分可以通用
    common_config = config["common_hp"]
    init_random_seed(common_config)
    device, rank, local_rank, world_size, node_num = init_ddp_info_params(already_init)

    train_conf = common_config['train_conf']
    feature_conf = common_config['feature_conf']
    learning_rate = train_conf['learning_rate']
    lr_scaling = train_conf['lr_scaling']
    export_conf = common_config['export_conf']
    data_loader_conf = common_config['data_loader_conf']
    model_conf = common_config['model_conf']
    dataset_name = data_loader_conf["dataset_name"]
    feature_conf["period"] = period
    train_conf['learning_rate'] = init_learning_rate(learning_rate, lr_scaling, world_size)
    filter_invalid_ids = train_conf.get('filter_invalid_ids', False)
    logging.info(f'debug filter invalid ids {filter_invalid_ids}')

    logging.info("Testing model on rank %s (local rank: %s); device: %s; world_size: %s;",
                 rank, local_rank, device, world_size)

    init_torch_config(local_rank, train_conf)

    # 1.2 这一部分根据业务数据格式，需要用特定的方式编辑feat_conf和model_conf

    feature_conf, model_conf, feature_map = refine_feat_and_model_conf(dataset=dataset_name,
                                                                       feature_conf=feature_conf,
                                                                       model_conf=model_conf,
                                                                       feature_map_dir_or_path=feature_map_dir_or_path,
                                                                       model_cfg=config[Const.MODEL_CFG])
    # 2. 初始化数据集
    dataset, eval_data_loader, _ = get_data_loaders(dataset=dataset_name,
                                                    data_dir=data_dir,
                                                    rank=rank,
                                                    world_size=world_size,
                                                    train_config=train_conf,
                                                    model_conf=model_conf,
                                                    feature_config=feature_conf,
                                                    dataloader_config=data_loader_conf)

    # 6.加载模型
    model_file_name = "model_hstu.pth"
    model_path = os.path.join(save_dir, export_conf["save_dir_name"], model_file_name)
    model_state_dict = torch.load(model_path, map_location=device)

    ModelRegistry.register_all_modules(
        modul_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "modeling/generic/sequential_v2")))
    model = ModelInitializer.init(gr_module_cfg=config)
    if hasattr(model, 'negative_sampler') and hasattr(model.negative_sampler, 'set_all_item_ids') and train_conf.get(
            'filter_invalid_ids', False):
        model.negative_sampler.set_all_item_ids(dataset.all_item_ids)

    model.load_state_dict(model_state_dict)

    evaluate_step(model=model,
                  feature_conf=feature_conf,
                  train_conf=train_conf,
                  world_size=world_size,
                  rank=rank,
                  save_after_eval=True,
                  device=device,
                  dataset=dataset_name,
                  all_item_ids=dataset.all_item_ids,
                  filter_invalid_ids=filter_invalid_ids)


def save_user_emb_fn(config, data_dir, save_dir, emb_dir, feature_max_index_path, period, feature_map_path) -> None:
    """
    单独评估函数
    """
    # 1. 初始化随机数、加速器，配置文件
    # 1.1 这一部分可以通用
    common_config = config["common_hp"]
    init_random_seed(common_config)
    if FLAGS.is_train:
        already_init = True
    else:
        already_init = False
    device, rank, local_rank, world_size, node_num = init_ddp_info_params(already_init)

    train_conf = common_config['train_conf']
    feature_conf = common_config['feature_conf']
    learning_rate = train_conf['learning_rate']
    lr_scaling = train_conf['lr_scaling']
    export_conf = common_config['export_conf']
    data_loader_conf = common_config['data_loader_conf']
    model_conf = common_config['model_conf']
    dataset_name = data_loader_conf["dataset_name"]
    feature_conf["period"] = period
    train_conf['learning_rate'] = init_learning_rate(learning_rate, lr_scaling, world_size)
    emb_field_delimiter = export_conf.get('emb_field_delimiter', '\t')
    emb_delimiter = export_conf.get('emb_delimiter', ',')
    save_item_embs_for_files = export_conf.get('save_item_embs_for_files', True)
    save_user_embs_for_files = export_conf.get('save_user_embs_for_files', True)
    feature_map_item_field_name = export_conf.get('feature_map_item_field_name', "itemid")
    feature_map_user_field_name = export_conf.get('feature_map_user_field_name', "upid")

    logging.info("Testing model on rank %s (local rank: %s); device: %s; world_size: %s;",
                 rank, local_rank, device, world_size)

    init_torch_config(local_rank, train_conf)

    # 1.2 这一部分根据业务数据格式，需要用特定的方式编辑feat_conf和model_conf

    feature_conf, model_conf, feature_map = refine_feat_and_model_conf(dataset=dataset_name,
                                                                       feature_conf=feature_conf,
                                                                       model_conf=model_conf,
                                                                       feature_map_dir_or_path=feature_max_index_path,
                                                                       model_cfg=config[Const.MODEL_CFG])
    # 2. 初始化数据集
    dataset, eval_data_loader, _ = get_data_loaders(dataset=dataset_name,
                                                    data_dir=data_dir,
                                                    rank=rank,
                                                    world_size=world_size,
                                                    train_config=train_conf,
                                                    model_conf=model_conf,
                                                    feature_config=feature_conf,
                                                    dataloader_config=data_loader_conf)

    # 6.加载模型
    model_file_name = "model_hstu.pth"
    model_path = os.path.join(save_dir, export_conf["save_dir_name"], model_file_name)
    model_state_dict = torch.load(model_path, map_location=device)

    ModelRegistry.register_all_modules(
        modul_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "modeling/generic/sequential_v2")))
    model = ModelInitializer.init(gr_module_cfg=config)
    if hasattr(model, 'negative_sampler') and hasattr(model.negative_sampler, 'set_all_item_ids') and train_conf.get(
            'filter_invalid_ids', False):
        model.negative_sampler.set_all_item_ids(dataset.all_item_ids)

    model.load_state_dict(model_state_dict)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    model.eval()
    if save_user_embs_for_files:
        save_emb_async_loader = mp.Process(target=async_data_loader,
                            args=(eval_data_loader, generate_emb_data_queue, True, 1))
        save_emb_async_loader.start()
        save_user_embs(model=model,
                       train_conf=train_conf,
                       feature_map_path=feature_map_path,
                       emb_dir=emb_dir,
                       rank=rank,
                       device=device,
                       emb_field_delimiter=emb_field_delimiter,
                       emb_delimiter=emb_delimiter,
                       feature_map_field_name=feature_map_user_field_name)

    if save_item_embs_for_files:
        save_item_embs(model=model,
                       feature_name=feature_conf.get("infer_items_key", "item_id"),
                       feature_map_path=feature_map_path,
                       emb_dir=emb_dir,
                       rank=rank,
                       device=device,
                       emb_field_delimiter=emb_field_delimiter,
                       emb_delimiter=emb_delimiter,
                       feature_map_field_name=feature_map_item_field_name)


def get_train_eval_dataloader(dataset, data_dir, rank, world_size, train_conf, model_conf, feature_conf,
                              dataloader_conf, data_pth='', is_recall=False):
    max_sequence_length = model_conf.get("max_sequence_length", 100)
    local_batch_size = train_conf.get("local_batch_size", 256)
    eval_batch_size = train_conf.get("eval_batch_size", 256)
    gr_output_length = train_conf.get("gr_output_length", 0)    
    prefetch_factor = dataloader_conf['prefetch_factor']
    num_workers = dataloader_conf['num_workers']

    dataset = get_reco_dataset(
        dataset=dataset,
        data_dir=data_dir,
        pth=data_pth,
        rank=rank,
        world_size=world_size,
        max_sequence_length=max_sequence_length,
        chronological=True,
        feature_conf=feature_conf,
        num_rerank=dataloader_conf.get("num_rerank", 256)
    )
    
    train_data_loader = create_data_loader(
        dataset.train_dataset,
        batch_size=local_batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers
    )
    eval_data_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=eval_batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers
    )
    return dataset, eval_data_loader, train_data_loader


def init_learning_rate(learning_rate, lr_scaling, world_size):
    lr_scaling = lr_scaling.strip().lower()
    if lr_scaling == "linear":
        learning_rate *= world_size
    elif lr_scaling == "sqrt":
        learning_rate *= sqrt(world_size)
    else:
        raise ValueError("'%s' is not a supported scaling strategy for the learning rate." % lr_scaling)
    return learning_rate


def main_torchrun(argv):
    if FLAGS.config_file is None:
        raise ValueError("you have to assign the train config file")
    config = get_config(FLAGS.config_file)

    if FLAGS.is_train:
        train_fn(config,
                 FLAGS.data_dir,
                 FLAGS.save_dir,
                 FLAGS.feature_map_dir,
                 str(FLAGS.period))
    else:
        eval_fn(config,
                FLAGS.data_dir,
                FLAGS.save_dir,
                FLAGS.feature_map_dir,
                str(FLAGS.period))

    if FLAGS.save_user_emb:
        save_user_emb_fn(config,
                         FLAGS.data_dir,
                         FLAGS.save_dir,
                         FLAGS.emb_dir,
                         FLAGS.feature_map_dir,
                         str(FLAGS.period),
                         FLAGS.feature_map_dir)


if __name__ == "__main__":
    app.run(main_torchrun)
