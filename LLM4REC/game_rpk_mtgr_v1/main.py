import collections
import copy
import datetime
import json
import logging
import os
import pickle
import random
import re
import sys
import time
from math import sqrt
from statistics import mean
from typing import Callable
from typing import Union, Iterable, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch_npu
from absl import app, flags
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch import optim
from torch.nn.functional import embedding

from const_global import myclass
from data.data_loader import create_data_loader
from data.eval import gather_all_dict, avg_eval, MetricsCalculator, \
    ag_process_rank_scores
from data.reco_dataset import get_reco_dataset
from modeling.generic.sequential.features import SequentialFeatures
from modeling.generic.sequential.local_trainer import assemble_model_executable_callback, \
    feed_datas_to_exec_callback, assemble_ag_model_executable_callback
from modeling.generic.utils.constants import Const
from modeling.model_initializer import ModelInitializer
from modeling.model_registry import ModelRegistry
from utils.common_utils import get_config, refine_feat_and_model_conf
from utils.model_saver import ModelSaver

# 初始化传入参数
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

flags.DEFINE_string("config_file", "./test.config", "Path to the config file.")
flags.DEFINE_integer("master_port", 12355, "Master port.")
flags.DEFINE_string("data_dir", "baseline_20251018", "Path to data.")
flags.DEFINE_string("save_dir", "baseline_20251018/output", "Path to save.")
flags.DEFINE_string("feature_map_dir", "baseline_20251018/config/feature_map.json",
                    "Path of feature_map.")
flags.DEFINE_string("feature_map_max_index_path", "baseline_20251018/config/feature_map.json",
                    "Path of feature map max index")
flags.DEFINE_string("llm_embedding_path", "baseline_20251018/llm_embedding_old",
                    "Path of pretrained llm embedding.")
flags.DEFINE_string("period", "20251018-000000", "Period of task execution.")
flags.DEFINE_boolean("is_train", True, "If the model is training or testing.")
flags.DEFINE_string("use_amp", "",
                    "Override use_amp from config: 'true' or 'false'. Empty string means use config value.")

FLAGS = flags.FLAGS


def init_ddp_info_params_mtp(already_init=False):
    """
    初始化加速器参数
    """

    addr = os.getenv("MASTER_ADDR")
    port = os.getenv("MASTER_PORT")
    rank_id = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_num = int(os.environ["NNODES"])
    logging.info(f"tcp://{addr}:{port}, nnodes={node_num}, rank={rank_id}")

    if not already_init:
        # initialize the process group
        dist.init_process_group("hccl", init_method=f"tcp://{addr}:{port}", rank=rank_id,
                                world_size=world_size)

    return f"npu:{local_rank}", rank_id, local_rank, world_size, node_num


def init_ddp_info_params(already_init=False):
    """
    初始化加速器参数
    """

    addr = os.getenv("MASTER_ADDR", "localhost")
    port = os.getenv("MASTER_PORT", "12356")

    rank_id = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    node_num = int(os.getenv("NNODES", "1"))  # 默认为 1 台机器

    logging.info(f"tcp://{addr}:{port}, nnodes={node_num}, rank={rank_id}")

    if not already_init:
        # initialize the process group
        dist.init_process_group("hccl", init_method=f"tcp://{addr}:{port}", rank=rank_id,
                                world_size=world_size)

    return f"npu:{local_rank}", rank_id, local_rank, world_size, node_num


def init_random_seed(config):
    """
    设置全局随机数以使训练结果更确定
    """
    random_seed = config.get('seed_conf', {}).get("global_seed", '1234')
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch_npu.npu.manual_seed(random_seed)
    torch_npu.npu.manual_seed_all(random_seed)


def init_torch_config(local_rank, train_conf, amp_dtype="fp16"):
    """
    设置torch配置
    """
    use_tf32 = train_conf.get("use_tf32", False)
    torch_npu.npu.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    use_amp = train_conf.get("use_amp", False)
    if use_amp:
        logging.info("AMP config: use_amp=%s, amp_dtype=%s (source: model_conf)", use_amp, amp_dtype)
        if amp_dtype == "bf16":
            logging.info("AMP (BF16) enabled on NPU. Loss scaling disabled (not needed for BF16).")
        else:
            logging.info("AMP (FP16) enabled on NPU. Dynamic loss scaling: %s, init scale: %s",
                         train_conf.get("dynamic_loss_scale", True),
                         train_conf.get("loss_scale", 2 ** 16))

    lr_schedule_type = train_conf.get("lr_schedule_type", "constant")
    if lr_schedule_type != "constant":
        logging.info("LR schedule: type=%s, warmup_steps=%d, total_steps=%d",
                     lr_schedule_type,
                     train_conf.get("num_warmup_steps", 0),
                     train_conf.get("total_training_steps", 0))


def get_data_loaders(dataset, data_dir, rank, world_size, train_config, model_conf, feature_config, dataloader_config):
    """
    获取训练和验证数据 dataloader
    """
    dataset, eval_data_loader, train_data_loader = get_train_eval_dataloader(
        dataset, data_dir, rank, world_size, train_config, model_conf, feature_config, dataloader_config
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


def get_optimizer_callback(train_conf) -> Tuple[Union[Iterable[Tensor], Iterable[Dict[str, Any]]], optim.Optimizer]:
    optimizer_type = train_conf["optimizer_type"].strip().lower()

    beta = tuple(train_conf["beta"])
    learning_rate = train_conf["learning_rate"]
    weight_decay = train_conf["weight_decay"]

    if optimizer_type == "adamw":
        return lambda params: torch.optim.AdamW(params, lr=learning_rate, betas=beta, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        return lambda params: torch.optim.Adam(params, lr=learning_rate, betas=beta, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return lambda params: torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer_type %s" % optimizer_type)


def get_optimizer_callback_group_lr(
        train_conf: Dict[str, Any],
        model: torch.nn.Module
) -> Tuple[Union[Iterable[Tensor], Iterable[Dict[str, Any]]], optim.Optimizer]:
    """创建优化器函数"""
    optimizer_type = train_conf["optimizer_type"].strip().lower()
    beta = tuple(train_conf["beta"]) if "beta" in train_conf else (0.9, 0.999)
    learning_rate = train_conf["learning_rate"]
    weight_decay = train_conf.get("weight_decay", 0.0)
    embedding_lr = train_conf.get('embedding_lr', learning_rate)
    dense_lr = train_conf.get('dense_lr', learning_rate)

    # 构建模型参数到名称的映射
    param_to_name = {param: name.lower() for name, param in model.named_parameters()}

    def optimizer_constructor(parameters):
        """
        优化器构造函数
        """
        embedding_params = []
        dense_params = []

        # 转换为列表确保可以多次迭代
        param_list = list(parameters) if isinstance(parameters, collections.abc.Iterable) else list(parameters)

        # 分类参数
        for param in param_list:
            # 检查参数是否需要优化
            if not getattr(param, 'requires_grad', False) or not getattr(param, 'is_leaf', True):
                continue

            # 查找参数名称
            param_name = param_to_name.get(param, "")

            if "embedding" in param_name:
                embedding_params.append(param)
            else:
                dense_params.append(param)
        # 构建参数组
        param_groups = []

        if dense_params:
            group = {'params': dense_params, 'lr': dense_lr}
            if weight_decay > 0:
                group['weight_decay'] = weight_decay
            param_groups.append(group)

        if embedding_params:
            group = {'params': embedding_params, 'lr': embedding_lr}
            if weight_decay > 0:
                group['weight_decay'] = weight_decay
            param_groups.append(group)
        # 创建优化器
        if optimizer_type == "adamw":
            return torch.optim.AdamW(param_groups, betas=beta)
        elif optimizer_type == "adam":
            return torch.optim.Adam(param_groups, betas=beta)
        elif optimizer_type == "sgd":
            return torch.optim.SGD(param_groups)
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    return optimizer_constructor


def train_step(model, exec_cb: Callable[[int], Tuple[torch.Tensor, SequentialFeatures]],
               train_conf, export_conf, batch_id, world_size, rank, epoch, device, save_dir,
               profiler=None):
    """
    训练模型. 每隔eval_interval步打印一次统计信息(耗时、loss),
    若exec_cb携带get_lr方法则同时打印当前学习率.
    若profiler不为None，则在profiler上下文中执行训练循环，每个step后调用prof.step().
    """
    model.train()
    train_costs, batch_group = [], train_conf['eval_interval']
    is_output_loss_csv_file = train_conf.get("is_output_loss_csv_file", False)
    step_list, loss_list = [], []

    last_training_time = time.time()

    def _run_one_step(batch_id):
        """执行单个训练step，返回loss。由profiler和非profiler路径共用。"""
        nonlocal last_training_time

        loss, _ = exec_cb(batch_id)

        if rank == 0 and is_output_loss_csv_file:
            loss_list.append(loss.detach().cpu().item())
            step_list.append(batch_id)

        if (batch_id % batch_group) == 0:
            # NPU异步执行：需synchronize确保所有算子执行完毕后再计时
            torch.npu.synchronize()
            train_cost = time.time() - last_training_time
            last_training_time = time.time()
            current_lr = exec_cb.get_lr() if hasattr(exec_cb, 'get_lr') else None
            if current_lr is not None:
                logging.info("rank %s; batch-stat (train): step %s "
                             "(epoch %s in %.2fs): loss=%.6f, lr=%.2e",
                             rank, batch_id, epoch, train_cost, loss, current_lr)
            else:
                logging.info("rank %s; batch-stat (train): step %s "
                             "(epoch %s in %.2fs): %.6f", rank, batch_id, epoch, train_cost, loss)
            if batch_id > 0:
                train_costs.append(train_cost)
        return loss

    if profiler is not None:
        with profiler as prof:
            while True:
                try:
                    loss = _run_one_step(batch_id)
                except StopIteration:
                    logging.info("finished")
                    break
                batch_id += 1
                prof.step()
    else:
        while True:
            try:
                loss = _run_one_step(batch_id)
            except StopIteration:
                logging.info("finished")
                break
            batch_id += 1

    if train_costs:
        logging.info(f"rank {rank} epoch {epoch}; mean cost per step: {mean(train_costs) / batch_group:.4f}s")
    else:
        logging.info(f"rank {rank} epoch {epoch}; less than {batch_group} steps")

    # 原来计算loss的方法
    epoch_loss = avg_eval(torch.tensor([loss]).to(device), world_size=world_size)

    # 训练完成，保存模型
    if rank == 0:
        if not os.path.exists(os.path.join(save_dir, export_conf["save_dir_name"])):
            os.makedirs(os.path.join(save_dir, export_conf["save_dir_name"]))
        save_model(model.module, save_dir, export_conf, epoch, train_conf)

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
    df.to_csv(loss_file_path, index=False, chunksize=100000)


def save_model(model,
               save_dir,
               export_conf,
               epoch,
               train_conf):
    """
    保存模型
    """
    if train_conf.get('phase', 'pretrain') == 'pretrain':  # 预训练阶段保存模型
        # 1. 创建一个统一的、有意义的输出目录
        checkpoint_dir = os.path.join(save_dir, export_conf["save_dir_name"])
        os.makedirs(checkpoint_dir, exist_ok=True)
        logging.info(f"Saving model artifacts to: {checkpoint_dir}")

        model.eval()

        if hasattr(model, 'embedding_module'):
            embedding_path = os.path.join(checkpoint_dir, "embedding_module.pth")
            torch.save(model.embedding_module.state_dict(), embedding_path)
            logging.info(f"Embedding module's state_dict saved to {embedding_path}")
        else:
            logging.warning("Model does not have 'embedding_module' attribute, skipping save.")

        if hasattr(model, 'sim_model'):
            sim_model_path = os.path.join(checkpoint_dir, "sim_model.pth")
            torch.save(model.sim_model.state_dict(), sim_model_path)
            logging.info(f"SIM model's state_dict saved to {sim_model_path}")
        else:
            logging.warning("Model does not have 'sim_model' attribute, skipping save.")

    else:
        if not os.path.exists(os.path.join(save_dir, export_conf["save_dir_name"])):
            os.makedirs(os.path.join(save_dir, export_conf["save_dir_name"]), exist_ok=True)

        post_name = ""
        is_pe = train_conf.get("add_pos_emb", False)
        is_rope = train_conf.get("add_rope_emb", False)
        is_fixed_pe = train_conf.get("add_fixed_pos_emb", False)

        if is_pe:
            post_name += "_pe"
        if is_rope:
            post_name += "_rope"
        if is_fixed_pe:
            post_name += "_fixed_pe"
        model_file_name = f"model_hstu{post_name}_{epoch}.pth"
        model_path = os.path.join(save_dir, export_conf["save_dir_name"], model_file_name)
        model.eval()
        logging.info("Saving model to %s", model_path)
        # 新版代码只支持保存state_dict
        torch.save(model.state_dict(), model_path)


def save_input_output(
        model,
        save_dir,
        model_input_export_list,
        export_conf,
        feature_conf,
        device,
        rank
):
    global dataset_name
    # ["fake", "real"]两种模式，即导出的候选集是真实数据还是随机生成的数据
    export_mode = export_conf.get("export_mode", "fake")

    model_input_export_list_new = []

    export_batch_size = export_conf.get('export_batch_size', 1)
    num_rerank = export_conf.get("num_rerank", 1)
    shape_checked = False

    cut_off_timestamp = feature_conf.get("cut_off_time", 0)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    logging.info("export_batch_size is %s", export_batch_size)
    logging.info("num_rerank is %s", num_rerank)

    timestamp_key = feature_conf.get("candidate_timestamps_column")
    date_key = feature_conf.get("candidate_date_column")
    time_feature_keys = {
        feature_conf.get("history_timestamps_column"),
        timestamp_key,
        feature_conf.get("history_date_column"),
        date_key,
        "timestamps",
    }
    time_feature_keys.discard(None)
    label_key = feature_conf.get("candidate_ratings_column")
    id_key = feature_conf.get("candidate_items_key")
    candidate_col_list = [
                             timestamp_key,
                             date_key,
                             label_key,
                             id_key] + [k for k in feature_conf["candidate_item_feature_columns"]] + [
                             "candidate_action_type"]
    candidate_col_list = list(set(candidate_col_list))
    for inputs in model_input_export_list:

        for k in inputs:
            inputs[k] = inputs[k][:export_batch_size]
            if export_mode == "real":
                if "candidate_" in k:
                    _, real_length = inputs[k].size()
                    inputs[k] = torch.nn.functional.pad(inputs[k], (0, num_rerank - real_length))
            elif export_mode == "fake":
                new_k = k
                if k in candidate_col_list:
                    if k == timestamp_key:
                        len_candidate_timestamps_fake = num_rerank * export_batch_size
                        candidate_timestamps_fake = [int(cut_off_timestamp + 86399)] * len_candidate_timestamps_fake
                        candidate_timestamps_fake = torch.tensor(candidate_timestamps_fake, dtype=torch.int64).view(
                            export_batch_size, -1).to(device)
                        inputs[k] = candidate_timestamps_fake
                    elif k == label_key or k == "candidate_ratings" or k == "candidate_action_type":
                        candidate_labels_fake = torch.randint(2, (export_batch_size, num_rerank), dtype=torch.int64).to(
                            device)
                        inputs[k] = candidate_labels_fake
                    elif k == date_key:
                        len_candidate_date_fake = num_rerank * export_batch_size
                        ts = cut_off_timestamp + 86399
                        dt = datetime.datetime.fromtimestamp(ts).date()
                        candidate_date_fake = [int(dt.strftime("%Y%m%d"))] * len_candidate_date_fake
                        candidate_date_fake = torch.tensor(candidate_date_fake, dtype=torch.int64).view(
                            export_batch_size, -1).to(device)
                        inputs[k] = candidate_date_fake
                    elif k == id_key:
                        candidate_feature_count = feature_conf["candidate_item_feature_columns"][id_key][
                            "feature_count"]
                        candidate_item_id_fake = torch.randint(candidate_feature_count, (export_batch_size, num_rerank),
                                                               dtype=torch.int64).to(device)
                        inputs[k] = candidate_item_id_fake
                    elif k in feature_conf["candidate_item_feature_columns"]:
                        candidate_dtype = feature_conf["candidate_item_feature_columns"][new_k]["dtype"]
                        if candidate_dtype == "int" or candidate_dtype == "context":
                            candidate_feature_count = feature_conf["candidate_item_feature_columns"][new_k][
                                "feature_count"]
                            candidate_feat_fake = torch.randint(candidate_feature_count,
                                                                (export_batch_size, num_rerank), dtype=torch.int64).to(
                                device)
                        elif candidate_dtype == "con":
                            candidate_feat_fake = torch.rand((export_batch_size, num_rerank), dtype=torch.float32).to(
                                device)
                        elif candidate_dtype == "multi":
                            candidate_feature_count = feature_conf["candidate_item_feature_columns"][new_k][
                                "feature_count"]
                            max_len = feature_conf["candidate_item_feature_columns"][new_k]["max_len"]
                            candidate_feat_fake = torch.randint(candidate_feature_count,
                                                                (export_batch_size, num_rerank, max_len),
                                                                dtype=torch.int64).to(device)
                        else:
                            candidate_feat_fake = torch.zeros((export_batch_size, num_rerank), dtype=torch.int64).to(
                                device)
                        inputs[k] = candidate_feat_fake

                elif k == "labels" or k == "loss_weights":
                    labels_fake = torch.randint(2, (export_batch_size, num_rerank), dtype=torch.int64).to(device)
                    inputs[k] = labels_fake

            else:
                logging.error("export_mode in export_conf should be either real or fake")

        if "history_timestamps" in inputs:  # 去掉history_timestamps
            print("found hist ts")
            inputs["timestamps"] = inputs["history_timestamps"]
        input_keys = list(inputs.keys())
        for k in input_keys:
            new_k = k
            if k in feature_conf["candidate_item_feature_columns"]:
                enabled = feature_conf["candidate_item_feature_columns"][new_k].get("enabled", True)
                if not enabled:
                    inputs.pop(k, None)
            elif k in feature_conf["history_item_feature_columns"]:
                enabled = feature_conf["history_item_feature_columns"][new_k].get("enabled", True)
                if not enabled:
                    inputs.pop(k, None)
            elif k in feature_conf["user_feature_columns"]:
                enabled = feature_conf["user_feature_columns"][k].get("enabled", True)
                if not enabled:
                    inputs.pop(k, None)
            # 以下这些key值不在音乐业务里使用
            elif k in ["candidate_item_id"] and dataset_name == "music-scalingraw-rank":
                inputs.pop(k, None)
        if not shape_checked and rank == 0:
            for k, v in inputs.items():
                logging.info("name: %s | shape: %s", k, to_numpy(v).shape)
            shape_checked = True
        model_input_export_list_new.append(inputs)
    infer_item_id_name = id_key  # "candidate_" + feature_conf["infer_items_key"]
    y_true_eval_all = []
    y_score_eval_all = []
    output_all = None
    input_all = {}
    input_keys = list(inputs.keys())
    candidate_label_csv = []
    score_csv = []
    for inputs in model_input_export_list_new:
        model_out = model(inputs, is_train=False)["rerank_score"]
        candidate_labels = inputs["labels"]
        candidate_ids = inputs[infer_item_id_name]
        y_true_eval = to_numpy(candidate_labels[candidate_ids != 0])
        y_score_eval = to_numpy(model_out[candidate_ids != 0])
        for candidate_label_row, candidate_id_row, model_out_row in zip(candidate_labels, candidate_ids, model_out):
            candidate_label_csv.append("^".join(str(x) for x in to_numpy(candidate_label_row[candidate_id_row != 0])))
            score_csv.append("^".join(str(x) for x in to_numpy(model_out_row[candidate_id_row != 0])))
        model_output = to_numpy(model_out)
        inputs = {
            k: to_numpy(inputs[k])
            for k in inputs
        }
        y_true_eval_all.append(y_true_eval)
        y_score_eval_all.append(y_score_eval)
        if output_all is None:
            output_all = model_output
        else:
            output_all = np.concatenate([output_all, model_output], axis=0)
        for k in input_keys:
            if k not in input_all:
                input_all[k] = inputs[k]
            else:
                input_all[k] = np.concatenate([input_all[k], inputs[k]], axis=0)

    if rank == 0:
        is_compute_export_auc = export_conf.get('compute_auc', False)
        if is_compute_export_auc:
            y_true_eval_all = np.concatenate(y_true_eval_all, axis=None)
            y_score_eval_all = np.concatenate(y_score_eval_all, axis=None)
            export_auc = roc_auc_score(y_true_eval_all, y_score_eval_all)
            logging.info("export auc is %s", export_auc)

        with open("%s/%s/pth_input" % (save_dir, export_conf["save_dir_name"]), 'wb') as fp:
            pickle.dump(input_all, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open("%s/%s/pth_output" % (save_dir, export_conf["save_dir_name"]), 'wb') as fp:
            pickle.dump(output_all, fp, protocol=pickle.HIGHEST_PROTOCOL)

        uid_csv = input_all.get("uid").tolist()
        need_export_csv = export_conf.get("need_export_csv", False)
        if need_export_csv:
            export_csv = pd.DataFrame({"column_id": uid_csv, "label_id": candidate_label_csv, "score": score_csv})
            with open("%s/%s/result.csv" % (save_dir, export_conf["save_dir_name"]), 'wb') as fp:
                export_csv.to_csv(fp, index=False)


def save_feature_map(export_conf, save_dir, feature_map_dir_or_path, feature_map=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists("%s/%s.config" % (save_dir, export_conf["save_dir_name"])):
        os.mkdir("%s/%s.config" % (save_dir, export_conf["save_dir_name"]))

    if feature_map is not None:
        feature_dict = feature_map
    else:
        feature_map_files = os.listdir(feature_map_dir_or_path)
        feature_dict = dict()
        for feature_map_file in feature_map_files:
            feature_map_df = pd.read_orc(os.path.join(feature_map_dir_or_path, feature_map_file))
            indices_to_remove = [i for i, value in enumerate(feature_map_df['feature_name']) if 'user_id' in value]
            feature_map_df = feature_map_df.drop(indices_to_remove)
            feature_name = feature_map_df['feature_name'].tolist()
            feature_id = feature_map_df['feature_id'].tolist()
            feature_id = [int(x) for x in feature_id]
            feature_dict.update(zip(feature_name, feature_id))

    with open("%s/%s.config/feature_map.json" % (save_dir, export_conf["save_dir_name"]), 'w', encoding="utf-8") as fp:
        json.dump(feature_dict, fp, indent=4, ensure_ascii=False)
    logging.info("Saved feature map to %s/%s.config/feature_map.json", save_dir, export_conf["save_dir_name"])


def evaluate_step(model, exec_cb: Callable[[int], Tuple[torch.Tensor, SequentialFeatures]],
                  feature_conf, feature_map, gauc_querys, train_conf, save_dir, feature_map_dir_or_path,
                  export_conf, world_size, rank, save_after_eval, device, model_saver: ModelSaver, data_dir=None):
    """
    评估模型
    """
    save_cols_key = 'save_score_csv_item_cols'
    model.eval()
    logging.info("rank %s starting evaluation...", rank)

    calculate_group_auc = train_conf.get('calculate_group_auc', True)

    if calculate_group_auc:
        save_score_csv_item_cols = feature_conf.get(save_cols_key, {"ctype": ""})
        save_score_csv_user_cols = feature_conf.get('save_score_csv_user_cols', {})
    else:
        save_score_csv_item_cols = feature_conf.get(save_cols_key, {})
        save_score_csv_user_cols = feature_conf.get('save_score_csv_user_cols', {})

    scores_all, ground_truth_all, eval_weights_all = [], [], []
    group_keys_all = {
        'uid': [],
        'scores': [],
        'ground_truth': []
    }
    for key in save_score_csv_item_cols:
        group_keys_all[key] = []
    for key in save_score_csv_user_cols:
        group_keys_all[key] = []

    eval_iter = 0
    last_eval_time = time.perf_counter()

    is_input_copied = 0
    model_input_export_list = []
    save_eval_input_output = export_conf.get(
        "save_input_output",
        save_after_eval or export_conf.get("export_num_batch", 0) > 0
    )

    while True:
        try:
            with torch.no_grad():
                scores, eval_seq_features = exec_cb(eval_iter)
        except StopIteration:
            logging.info("finished")
            break

        if calculate_group_auc:
            scores, ground_truth, group_keys, model_input = ag_process_rank_scores(
                scores,
                eval_seq_features, feature_conf.get(save_cols_key, {"ctype": ""}),
                feature_conf.get('save_score_csv_user_cols', {})
            )
        else:
            # 评估函数由于不同业务要求不同，故需要不同业务各自实现。
            scores, ground_truth, group_keys, model_input = ag_process_rank_scores(
                scores, eval_seq_features, feature_conf.get(save_cols_key, {}),
                feature_conf.get('save_score_csv_user_cols', {})
            )

        scores_all.extend(scores.view(-1).detach().cpu().tolist())
        ground_truth_all.extend(ground_truth.view(-1).detach().cpu().tolist())
        group_keys_all["scores"].extend(scores.view(-1).detach().cpu().tolist())
        group_keys_all["ground_truth"].extend(ground_truth.view(-1).detach().cpu().tolist())
        for key in group_keys_all.keys():
            if key not in ["scores", "ground_truth"]:
                items = group_keys.get(key, torch.tensor([]))
                group_keys_all[key].extend(items.detach().view(-1).cpu().tolist())

        if (eval_iter % train_conf["eval_interval"]) == 0:
            torch.distributed.barrier()  # sync time taken
            cost = time.perf_counter() - last_eval_time
            logging.info(
                f"rank {rank}; batch-stat (eval): step {eval_iter} (EVAL in {cost:.2f}s)")
            last_eval_time = time.perf_counter()

        export_num_batch = export_conf.get("export_num_batch", 1)
        if save_eval_input_output and is_input_copied < export_num_batch:
            model_input_export = copy.deepcopy(model_input)
            model_input_export_list.append(model_input_export)
            is_input_copied += 1

        eval_iter += 1
        del scores
        del ground_truth
        del group_keys

    torch_npu.npu.empty_cache()
    logging.info("rank %s start gather...", rank)
    logging.info("rank %s gathering ground_truth_all, size: %s", rank, len(ground_truth_all))

    group_keys_all = gather_all_dict(group_keys_all, world_size=world_size)

    logging.info(
        'Number of samples in eval dataset: %s, rank %s, number of positive samples %s, '
        'number of negative samples %s',
        len(group_keys_all["scores"]), rank, sum(group_keys_all["ground_truth"]),
        len(group_keys_all["ground_truth"]) - sum(group_keys_all["ground_truth"]))

    if save_eval_input_output and is_input_copied == 0:
        logging.error("No input sample can be saved because all the candidates are zero.")
    save_result_to_local = feature_conf.get("save_result_to_local", True)

    if rank == 0:
        save_path = "%s/modelfile" % save_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        logging.info("Test data count: %d", len(group_keys_all["ground_truth"]))
        metrics_file_name = "metrics_report.csv"
        metrics_path = os.path.join(save_path, metrics_file_name)
        if save_result_to_local:
            logging.info("Saving metrics to %s", metrics_path)

        for k, v in group_keys_all.items():
            logging.info("group_keys_all[%s]: %s", k, len(v))

        eval_df = pd.DataFrame({
            **group_keys_all,
        })
        calculator = MetricsCalculator(eval_df, feature_map, gauc_querys, feature_conf.get('metric_list', []),
                                       feature_conf.get('group_metric_cols_values', []),
                                       feature_conf.get('bias_evaluation_conf', []), save_result_to_local,
                                       calculate_group_auc)
        eval_result = calculator.calculate(metrics_path)
        auc = eval_result.get("global").get("auc")
        if auc is None:
            logging.warning("AUC is None (likely NaN scores), skipping model save")
        elif auc > model_saver.best_auc:
            model_saver.best_auc = auc
            logging.info("Update best AUC: %f, save model", model_saver.best_auc)

            model_saver.save_model(model, "model_hstu.pth")
            model_saver.save_metric()

    if save_eval_input_output and model_input_export_list:
        save_dir_export = save_dir
        if not os.path.exists(save_dir_export):
            os.mkdir(save_dir_export)
        save_input_output(model.module, save_dir_export, model_input_export_list, export_conf,
                          feature_conf, device, rank)
    logging.info(
        'Number of samples in eval dataset: %s, rank %s, number of positive samples %s, '
        'number of negative samples %s',
        len(group_keys_all["scores"]), rank, sum(group_keys_all["ground_truth"]),
        len(group_keys_all["ground_truth"]) - sum(group_keys_all["ground_truth"]))

    return


def _model_size_in_gib(model):
    for name, param in model.named_parameters():
        print(f"层名称: {name} | 形状: {param.shape} | 参数量: {param.numel()}")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model total parameters: {total_params:.2f}M")
    emb_params = sum(p.numel() for n, p in model.named_parameters() if "embedding_module" in n) / 1e6
    print(f"Model emb. parameters: {emb_params:.2f}M")
    dlrm_params = sum(p.numel() for n, p in model.named_parameters() if "dlrm_module" in n) / 1e6
    print(f"Model dlrm parameters: {dlrm_params:.2f}M")
    rankmixer_params = sum(p.numel() for n, p in model.named_parameters() if "rankmixer" in n) / 1e6
    print(f"Model rankmixer parameters: {rankmixer_params:.2f}M")
    transformer_params = sum(p.numel() for n, p in model.named_parameters() if "sequence_model" in n) / 1e6
    print(f"Model transformer parameters: {transformer_params:.2f}M")

    other_params = 0
    for n, p in model.named_parameters():
        pararm_cond = ("embedding_module" not in n and
                       "dlrm_module" not in n and
                       "rankmixer" not in n and
                       "sequence_model" not in n)
        if pararm_cond:
            other_params += p.numel()
    other_params /= 1e6
    print(f"Model other(inp&ffn) parameters: {other_params:.2f}M")
    total_dense_params = dlrm_params + rankmixer_params + transformer_params + other_params
    print(f"Model total dense parameters: {total_dense_params:.2f}M")


def train_fn(config, data_dir, save_dir, feature_map_dir_or_path, llm_embedding_path,
             period, model_saver: ModelSaver) -> None:
    """
    训练函数

    :param config: 配置文件字典
    :param data_dir: 数据集路径
    :param save_dir: 模型保存路径
    :param feature_map_dir_or_path: feature_map文件路径
    :param llm_embedding_path: llm embedding文件路径
    :param period: 训练周期
    :param model_saver: 用于模型保存
    :return:
    """
    # 1. 初始化随机数、加速器，配置文件
    # 1.1 这一部分可以通用
    common_config = config[Const.COMMON_HP]
    init_random_seed(common_config)
    device, rank, local_rank, world_size, node_num = init_ddp_info_params()

    train_conf = common_config['train_conf']
    # 环境变量/命令行覆盖AMP配置
    _use_amp_override = FLAGS.use_amp or os.getenv("USE_AMP", "")
    if _use_amp_override:
        train_conf["use_amp"] = _use_amp_override.lower() in ("true", "1", "yes")
        logging.info("AMP config overridden: use_amp=%s", train_conf["use_amp"])
    feature_conf = common_config['feature_conf']
    learning_rate = train_conf['learning_rate']
    lr_scaling = train_conf['lr_scaling']
    export_conf = common_config['export_conf']
    data_loader_conf = common_config['data_loader_conf']
    model_conf = common_config['model_conf']
    amp_dtype = model_conf.get("amp_dtype", "fp16")
    dataset_name = data_loader_conf["dataset_name"]
    # 新增阶段：sim 的 pretrain 阶段 以及 hstu 的 train 阶段
    phase = train_conf.get('phase', 'pretrain')  # pretrain or train
    model_conf['phase'] = phase
    feature_conf['phase'] = phase
    export_conf['phase'] = phase

    use_group_lr = model_conf.get("use_group_lr", True)

    # 为了去读llm embedding文件方便，添加以下字段
    feature_conf['llm_embedding_file_path'] = llm_embedding_path
    feature_conf['feature_map_path'] = feature_map_dir_or_path

    train_conf['save_dir'] = save_dir  # 方便读取权重
    feature_conf["period"] = period
    train_conf['learning_rate'] = init_learning_rate(learning_rate, lr_scaling, world_size)

    # jump_eval_during_train = train_conf.get("jump_eval_during_train", True) # 跳过eval，没有发现为什么history timestamps 出问题了t
    if dataset_name == "ag-rank" or dataset_name == "Longer_dataset_ulan" or dataset_name == "Game":
        myclass.set_path(rootdir=save_dir, filename="analysis.csv")

    logging.info("Training model on rank %s (local rank: %s); device: %s; world_size: %s;",
                 rank, local_rank, device, world_size)

    init_torch_config(local_rank, train_conf, amp_dtype=amp_dtype)

    # 1.2 这一部分根据业务数据格式，需要用特定的方式编辑feat_conf和model_conf
    feature_conf, model_conf, feature_map, gauc_querys = refine_feat_and_model_conf(
        dataset=dataset_name,
        feature_conf=feature_conf,
        model_conf=model_conf,
        feature_map_dir_or_path=feature_map_dir_or_path,
        model_cfg=config[Const.MODEL_CFG],
    )

    # # 1.3 保存feature_map供后续模型调用
    if rank == 0:
        save_feature_map(export_conf=export_conf,
                         save_dir=save_dir,
                         feature_map_dir_or_path=feature_map_dir_or_path,
                         feature_map=feature_map)
        with open(os.path.join(save_dir, "gr_module_config.json"), "w", encoding="utf-8") as f:
            json.dump({"gr_module_config": config}, f, ensure_ascii=False, indent=4)

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
        modul_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "modeling/generic/sequential"))
    )
    model = ModelInitializer.init(gr_module_cfg=config)
    if rank == 0:
        _model_size_in_gib(model)

    # 4. 初始化优化器
    if use_group_lr:
        opt_cb = get_optimizer_callback_group_lr(train_conf, model)
    else:
        opt_cb = get_optimizer_callback(train_conf)

    """
    如果模型中确实存在未使用的参数（例如某些分支仅在特定条件下执行），可以通过设置 find_unused_parameters=True 来解决
    注意：启用此选项可能会略微降低性能
    """
    if dataset_name == "ag-rank" or dataset_name == "Longer_dataset_ulan" or dataset_name == "Game":
        find_unused_parameters = True
    else:
        find_unused_parameters = False

    if model_conf.get("root_model_type") == "GRModelEp":
        pass
    else:
        if dataset_name != "ag-rank" and dataset_name != "Longer_dataset_ulan" and dataset_name != "Game":
            model, exec_cb = assemble_model_executable_callback(
                model,
                opt_cb,
                local_rank,
                device,
                train_conf,
                feature_conf,
                find_unused_parameters,
                amp_dtype=amp_dtype,
            )
        else:
            model, exec_cb = assemble_ag_model_executable_callback(
                model,
                opt_cb,
                local_rank,
                device,
                train_conf,
                feature_conf,
                find_unused_parameters,
                amp_dtype=amp_dtype,
            )
        feed_func = feed_datas_to_exec_callback

    # NPU Profiler：按需开启性能数据采集
    use_profiler = train_conf.get("use_profiler", False)
    profiler = None
    if use_profiler:
        profiler_output_dir = train_conf.get("profiler_output_dir", "./profiler_result")
        record_shapes = train_conf.get("profiler_record_shapes", False)
        profile_memory = train_conf.get("profiler_profile_memory", False)
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Db,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False
        )
        profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=2, active=1, repeat=1, skip_first=3),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiler_output_dir),
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=True,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config)
        logging.info("NPU Profiler enabled: output_dir=%s, record_shapes=%s, profile_memory=%s",
                     profiler_output_dir, record_shapes, profile_memory)

    batch_id = 0
    for epoch in range(train_conf['num_epochs']):
        train_exec_cb, eval_exec_cb = feed_func(exec_cb, train_data_loader, eval_data_loader)
        train_step_para = {
            "model": model,
            "exec_cb": train_exec_cb,
            "train_conf": train_conf,
            "export_conf": export_conf,
            "batch_id": batch_id,
            "world_size": world_size,
            "rank": rank,
            "epoch": epoch,
            "device": device,
            "save_dir": save_dir,
            "profiler": profiler
        }

        batch_id, epoch_loss = train_step(**train_step_para)

        logging.info("loss at epoch %s is %s", epoch, epoch_loss)

        if dataset_name == "ag-rank" or dataset_name == "Longer_dataset_ulan" or dataset_name == "Game":
            # 保存统计结果
            logging.info("saving count results")
            myclass.save_data(rank)
        # 6. 测试模型
        evaluate_step(model=model,
                      exec_cb=eval_exec_cb,
                      feature_conf=feature_conf,
                      feature_map=feature_map,
                      gauc_querys=gauc_querys,
                      train_conf=train_conf,
                      save_dir=save_dir,
                      feature_map_dir_or_path=feature_map_dir_or_path,
                      export_conf=export_conf,
                      world_size=world_size,
                      rank=rank,
                      save_after_eval=train_conf.get("save_after_eval", False),
                      device=device,
                      model_saver=model_saver)

    del model
    del opt_cb
    del train_data_loader


def parse_online_test_case_json(input_json, output_csv_path):  # 将中文映射表直接存到了数据目录下面了
    def extract_json(s):
        match = re.search(r'\[(.*?)\]', s)

        result_list = []
        if match:
            content_str = match.group(1).strip()

            if content_str:
                number_strings = [item.strip() for item in content_str.split(',')]

            for item_str in number_strings:
                try:
                    # 尝试转换为整数
                    converted_item = int(item_str)
                except ValueError:
                    try:
                        # 再次尝试转换为浮点数
                        converted_item = float(item_str)
                    except ValueError:
                        # 如果两种数字转换都失败，就保留它作为原始字符串
                        converted_item = item_str

                result_list.append(converted_item)

            return result_list
        else:
            return result_list

    def parse_features(raw: str):
        """
        将形如 'key:{"k1":[1,2],"k2":[[1,2]]}' 的字符串解析成 Python dict。
        """
        # 找到第一个左大括号
        idx = raw.find('{')
        if idx == -1:
            raise ValueError(f"无法在特征字符串中找到 JSON 对象：{raw!r}")
        json_part = raw[idx:]
        return json.loads(json_part)

    # 1. 读取 JSON 文件
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}", file=sys.stderr)
        raise

    # 2. 读取 result -> respData -> recommendResult
    result = (
        data.get('result', {})
        .get('respData', {})[0]
        .get('recommendResult')
    )
    appid2ch = {}
    cc = result['content']

    for c in cc:
        appid2ch[c['appId']] = c['appName']
    # 3. 读取 deviceIDForFeature
    device_id = (
        data.get('interpretativeContext', {})
        .get('curContexts')
        .get('deviceIDForFeature')
    )
    # 4. 读取特征 dict：traceLog -> singleInvocationParams
    for cc in data.get('traceLog', {}):
        if "candidate_ubr_uninstall_app_list" in cc:
            raw_features = cc
            break
    try:
        if isinstance(raw_features, str):
            features = parse_features(raw_features)
        elif isinstance(raw_features, dict):
            features = raw_features
        else:
            raise TypeError(f"Unexpected type for features: {type(raw_features)}")
    except Exception as e:
        print(f"Error parsing features: {e}", file=sys.stderr)
        features = {}

    features['device_id_sha256'] = device_id

    # # 5. 读取 candidate_list
    for cc in data.get('traceLog', {}):
        if "candidate list is :" in cc:
            candidate_list = cc
            break

    # 6. 保存特征为 CSV
    # 将 features（一个 dict）转换为 DataFrame 的一行，然后写入 CSV
    try:
        df = pd.DataFrame([features])
        df.to_csv(os.path.join(output_csv_path, "online_test.csv"), index=False, encoding='utf-8-sig', sep=';')
        print(f"Features saved to {os.path.join(output_csv_path, 'online_test.csv')}")
    except Exception as e:
        print(f"Error saving CSV: {e}", file=sys.stderr)
    print(features.get('candidate_item_id'))
    # 7. 打印 device_id、candidate_list 及 result
    print("\n===== Extracted Information =====")
    print(f"Device ID: {device_id}")
    print(f"Candidate List: {candidate_list}")
    app_ids_items = extract_json(candidate_list)

    trans_dict = {}

    for item_id, app_id in zip(features.get('candidate_item_id'), app_ids_items):
        trans_dict[item_id] = {
            "app_id": app_id,
            "ch_name": appid2ch.get(app_id),
        }
    f = open(os.path.join(output_csv_path, "appid2ch.json"), "w")
    json.dump(appid2ch, f)

    f = open(os.path.join(output_csv_path, "app_ids.json"), "w")
    json.dump(app_ids_items, f)


def eval_fn(config, data_dir, save_dir, feature_map_dir_or_path, llm_embedding_path, period,
            model_saver: ModelSaver, already_init=False) -> None:
    """
    单独评估函数
    """
    # 1. 初始化随机数、加速器，配置文件
    # 1.1 这一部分可以通用
    common_config = config[Const.COMMON_HP]
    init_random_seed(common_config)
    device, rank, local_rank, world_size, node_num = init_ddp_info_params(already_init)

    train_conf = common_config['train_conf']
    # 环境变量/命令行覆盖AMP配置
    _use_amp_override = FLAGS.use_amp or os.getenv("USE_AMP", "")
    if _use_amp_override:
        train_conf["use_amp"] = _use_amp_override.lower() in ("true", "1", "yes")
        logging.info("AMP config overridden: use_amp=%s", train_conf["use_amp"])
    feature_conf = common_config['feature_conf']
    learning_rate = train_conf['learning_rate']
    lr_scaling = train_conf['lr_scaling']
    export_conf = common_config['export_conf']
    data_loader_conf = common_config['data_loader_conf']
    model_conf = common_config['model_conf']
    amp_dtype = model_conf.get("amp_dtype", "fp16")
    dataset_name = data_loader_conf["dataset_name"]
    feature_conf["period"] = period
    train_conf['learning_rate'] = init_learning_rate(learning_rate, lr_scaling, world_size)

    # 为了去读llm embedding文件方便，添加以下字段
    feature_conf['llm_embedding_file_path'] = llm_embedding_path
    feature_conf['feature_map_path'] = feature_map_dir_or_path

    logging.info("Testing model on rank %s (local rank: %s); device: %s; world_size: %s;",
                 rank, local_rank, device, world_size)

    init_torch_config(local_rank, train_conf, amp_dtype=amp_dtype)
    if train_conf.get("test_json", False):
        parse_online_test_case_json(train_conf['test_case_json_path'], data_dir)
        print("online_test_csv_process_complete!")

    load_ano_hstu = train_conf.get('load_ano_hstu', False)
    load_ano_hstu_path = train_conf.get('load_ano_hstu_path', "")
    # 1.2 这一部分根据业务数据格式，需要用特定的方式编辑feat_conf和model_conf

    feature_conf, model_conf, feature_map, gauc_querys = refine_feat_and_model_conf(
        dataset=dataset_name,
        feature_conf=feature_conf,
        model_conf=model_conf,
        feature_map_dir_or_path=feature_map_dir_or_path,
        model_cfg=config[Const.MODEL_CFG],
    )
    # 2. 初始化数据集
    dataset, eval_data_loader, train_data_loader = get_data_loaders(dataset=dataset_name,
                                                                    data_dir=data_dir,
                                                                    rank=rank,
                                                                    world_size=world_size,
                                                                    train_config=train_conf,
                                                                    model_conf=model_conf,
                                                                    feature_config=feature_conf,
                                                                    dataloader_config=data_loader_conf)

    # 6.加载模型
    ModelRegistry.register_all_modules(
        modul_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "modeling/generic/sequential")))
    model = ModelInitializer.init(gr_module_cfg=config)
    # 2. 根据 phase 决定加载策略
    phase = train_conf.get('phase', 'pretrain')
    checkpoint_dir = os.path.join(save_dir, export_conf["save_dir_name"])

    if phase == 'pretrain':
        # --- 预训练模式加载逻辑 ---
        logging.info(f"Loading model from pretrain phase artifacts in: {checkpoint_dir}")

        # 加载 embedding_module
        if hasattr(model, 'embedding_module'):
            embedding_path = os.path.join(checkpoint_dir, "embedding_module.pth")
            if os.path.exists(embedding_path):
                logging.info(f"Loading embedding module from {embedding_path}")
                state_dict = torch.load(embedding_path, map_location=device)
                model.embedding_module.load_state_dict(state_dict)
            else:
                logging.warning(f"Embedding module file not found: {embedding_path}")
        else:
            logging.warning("Model does not have 'embedding_module' attribute, skipping load.")

        # 加载 sim_model
        if hasattr(model, 'sim_model'):
            sim_model_path = os.path.join(checkpoint_dir, "sim_model.pth")
            if os.path.exists(sim_model_path):
                logging.info(f"Loading SIM model from {sim_model_path}")
                state_dict = torch.load(sim_model_path, map_location=device)
                model.sim_model.load_state_dict(state_dict)
            else:
                logging.warning(f"SIM model file not found: {sim_model_path}")
        else:
            logging.warning("Model does not have 'sim_model' attribute, skipping load.")

    else:
        # --- 微调/下游任务模式加载逻辑 --- train 和 test_case 都是加载
        # 构造与 save_model 完全一致的 post_name
        if load_ano_hstu:
            model_path = load_ano_hstu_path
        else:
            post_name = ""
            is_pe = train_conf.get("add_pos_emb", False)
            is_rope = train_conf.get("add_rope_emb", False)
            is_fixed_pe = train_conf.get("add_fixed_pos_emb", False)
            load_epoch = train_conf.get("load_epoch", 0)
            if is_pe:
                post_name += "_pe"
            if is_rope:
                post_name += "_rope"
            if is_fixed_pe:
                post_name += "_fixed_pe"

            # 构造与 save_model 完全一致的文件名
            model_file_name = f"model_hstu{post_name}.pth"
            model_path = os.path.join(checkpoint_dir, model_file_name)

        logging.info(f"Attempting to load model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # 加载整个模型的 state_dict
        model_state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state_dict)
        logging.info("Successfully loaded model state_dict.")
    if model_conf.get("root_model_type") == "GRModelEp":
        pass
    else:
        if dataset_name != "ag-rank" and dataset_name != "Longer_dataset_ulan" and dataset_name != "Game":
            model, exec_cb = assemble_model_executable_callback(
                model,
                None,
                local_rank,
                device,
                train_conf,
                feature_conf,
                amp_dtype=amp_dtype,
            )
        else:
            model, exec_cb = assemble_ag_model_executable_callback(
                model,
                None,
                local_rank,
                device,
                train_conf,
                feature_conf,
                amp_dtype=amp_dtype,
            )
        feed_func = feed_datas_to_exec_callback

    _, eval_exec_cb = feed_func(exec_cb, None, eval_data_loader)

    if dataset_name == "ag-rank" or dataset_name == "Longer_dataset_ulan" or dataset_name == "Game":
        # 保存统计结果
        myclass.save_data(rank)
    # 6. 测试模型
    evaluate_step(model=model,
                  exec_cb=eval_exec_cb,
                  feature_conf=feature_conf,
                  feature_map=feature_map,
                  gauc_querys=gauc_querys,
                  train_conf=train_conf,
                  save_dir=save_dir,
                  feature_map_dir_or_path=feature_map_dir_or_path,
                  export_conf=export_conf,
                  world_size=world_size,
                  rank=rank,
                  save_after_eval=train_conf.get("save_after_eval", False),
                  device=device,
                  data_dir=data_dir,
                  model_saver=model_saver)


def get_train_eval_dataloader(dataset, data_dir, rank, world_size, train_conf, model_conf, feature_conf,
                              dataloader_conf):
    max_sequence_length = model_conf.get("max_sequence_length", 100)
    local_batch_size = train_conf.get("local_batch_size", 256)
    eval_batch_size = train_conf.get("eval_batch_size", 256)
    gr_output_length = train_conf.get("gr_output_length", 0)
    prefetch_factor = dataloader_conf['prefetch_factor']
    num_workers = dataloader_conf['num_workers']
    train_data_path = dataloader_conf['train_data_path']
    valid_data_path = dataloader_conf['valid_data_path']
    use_dynamic_padding = dataloader_conf.get("use_dynamic_padding", False)
    padding_bucket_size = dataloader_conf.get("padding_bucket_size", 0)
    padding_bucket_table = dataloader_conf.get("padding_bucket_table", None)
    padding_side_origin = dataloader_conf.get("padding_side_origin", "left")
    padding_side = dataloader_conf.get("padding_side", padding_side_origin)

    dataset = get_reco_dataset(
        dataset=dataset,
        data_dir=data_dir,
        train_data_path=train_data_path,
        valid_data_path=valid_data_path,
        rank=rank,
        world_size=world_size,
        max_sequence_length=max_sequence_length,
        chronological=feature_conf.get("chronological", True),
        feature_conf=feature_conf,
        num_rerank=dataloader_conf.get("num_rerank", 256),
        history_length=dataloader_conf.get("history_length", 400),
        use_dynamic_padding=use_dynamic_padding,
        padding_side_origin=padding_side_origin,
        padding_side=padding_side
    )
    if model_conf.get("root_model_type") == "GRModelEp":
        from data.data_loader import create_data_loader_ep
        train_data_loader = create_data_loader_ep(
            dataset.train_dataset,
            batch_size=local_batch_size,
            item_feature_columns=feature_conf['item_feature_columns'],
            user_feature_columns=feature_conf['user_feature_columns'],
            max_output_length=gr_output_length,
            itemid_column_name=feature_conf['infer_items_key'],
            infer_ratings_key=feature_conf['infer_ratings_key'],
            infer_timestamps_key=feature_conf['infer_timestamps_key'],
            multi_value_prefix=feature_conf.get('multi_value_prefix', "pref_"),
            include_loss_weights=True,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers
        )
        eval_data_loader = create_data_loader_ep(
            dataset.eval_dataset,
            batch_size=eval_batch_size,
            item_feature_columns=feature_conf['item_feature_columns'],
            user_feature_columns=feature_conf['user_feature_columns'],
            max_output_length=gr_output_length + 1,
            itemid_column_name=feature_conf['infer_items_key'],
            infer_ratings_key=feature_conf['infer_ratings_key'],
            infer_timestamps_key=feature_conf['infer_timestamps_key'],
            multi_value_prefix=feature_conf.get('multi_value_prefix', "pref_"),
            include_loss_weights=True,
            include_candidate_items=True,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers
        )
    else:
        train_data_loader = create_data_loader(
            dataset.train_dataset,
            batch_size=local_batch_size,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers,
            use_dynamic_padding=use_dynamic_padding,
            padding_bucket_size=padding_bucket_size,
            padding_bucket_table=padding_bucket_table,
            padding_side=padding_side
        )
        eval_data_loader = create_data_loader(
            dataset.eval_dataset,
            batch_size=eval_batch_size,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers,
            use_dynamic_padding=use_dynamic_padding,
            padding_bucket_size=padding_bucket_size,
            padding_bucket_table=padding_bucket_table,
            padding_side=padding_side
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
    config = config["gr_module_config"]

    export_save_dir_name = config[Const.COMMON_HP]['export_conf']['save_dir_name']
    model_saver = ModelSaver(FLAGS.save_dir, export_save_dir_name)

    if FLAGS.is_train:
        train_fn(config,
                 FLAGS.data_dir,
                 FLAGS.save_dir,
                 FLAGS.feature_map_dir,
                 FLAGS.llm_embedding_path,
                 str(FLAGS.period),
                 model_saver)
    else:
        eval_fn(config,
                FLAGS.data_dir,
                FLAGS.save_dir,
                FLAGS.feature_map_dir,
                FLAGS.llm_embedding_path,
                str(FLAGS.period),
                model_saver)


if __name__ == "__main__":
    app.run(main_torchrun)
