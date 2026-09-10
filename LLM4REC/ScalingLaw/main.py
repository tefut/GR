import copy
import json
import logging
import os
import shutil
import stat
import pickle
import random
import sys
import time
from math import sqrt
from statistics import mean

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch_npu
from absl import app, flags

from const_global import myclass
from data.data_loader import create_data_loader
from data.eval import gather_all_list, avg_eval, MetricsCalculator
from data.reco_dataset import get_reco_dataset
from modeling.generic.executors.executor import Executor
from modeling.generic.executors.local_executor import LocalExecutor
from modeling.generic.utils.constants import Const
from modeling.model_initializer import ModelInitializer
from modeling.model_registry import ModelRegistry
from utils.common_utils import get_config, refine_feat_and_model_conf
from torch.distributed._shard.sharded_tensor import ShardedTensor

# 初始化传入参数
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

flags.DEFINE_string("config_file", None, "Path to the config file.")
flags.DEFINE_integer("master_port", 12355, "Master port.")
flags.DEFINE_string("data_dir", None, "Path to data.")
flags.DEFINE_string("save_dir", None, "Path to save.")
flags.DEFINE_string("feature_map_dir", None, "Path of feature_map.")
flags.DEFINE_string("feature_map_max_index_path", None, "Path of feature map max index")
flags.DEFINE_string("LLM_embedding_dir", None, "Path of LLM embeddings")
flags.DEFINE_string("period", None, "Period of task execution.")
flags.DEFINE_boolean("is_train", True, "If the model is training or testing.")

FLAGS = flags.FLAGS


def init_ddp_info_params(already_init=False):
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


def init_torch_config(local_rank, train_conf):
    """
    设置torch配置
    """
    use_tf32 = train_conf.get("use_tf32", False)
    torch_npu.npu.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    logging.info("cuda.matmul.allow_tf32: %s", use_tf32)
    logging.info("cudnn.allow_tf32: %s", use_tf32)


def get_data_loaders(dataset, data_dir, rank, world_size, train_config, model_conf, feature_config, dataloader_config):
    """
    获取训练和验证数据 dataloader
    """
    max_sequence_length = model_conf.get("max_sequence_length", 100)
    local_batch_size = train_config.get("local_batch_size", 256)
    eval_batch_size = train_config.get("eval_batch_size", 256)
    dataset, eval_data_loader, train_data_loader = get_train_eval_dataloader(
        dataset, data_dir, local_batch_size, eval_batch_size, max_sequence_length, rank, world_size, feature_config,
        dataloader_config
    )
    return dataset, eval_data_loader, train_data_loader


def train_step(executor: Executor, train_conf, export_conf, 
               batch_id, world_size, rank, epoch, device, save_dir, local_rank):
    """
    训练模型
    """
    train_costs, batch_group = [], train_conf['eval_interval']
    last_training_time = time.time_ns()
    is_output_loss_csv_file = train_conf.get("is_output_loss_csv_file", False)
    step_list, loss_list, grad_list = [], [], {}
    while True:
        try:
            loss, _ = executor.execute(batch_id)
        except StopIteration:
            logging.info("finished")
            break

        loss_list.append(loss.detach().cpu().item())
        step_list.append(batch_id)


        if (batch_id % batch_group) == 0:
            train_cost = time.time_ns() - last_training_time
            logging.info("rank %s; batch-stat (train): step %s "
                         "(epoch %s in %.2fs): %.6f", rank, batch_id, 
                         epoch, train_cost / 1000000000, mean(loss_list[-batch_group:]))
            if batch_id > 0:
                train_costs.append(train_cost)

            embs_grad, transformer_grad = [], []
            if is_output_loss_csv_file:
                for name, parm in executor.model.module.named_parameters():
                    if "embs" in name:
                        embs_grad.append(torch.norm(parm.grad, p=2).item())
                    elif "transformer" in name:
                        transformer_grad.append(torch.norm(parm.grad, p=2).item())
                grad_list.setdefault("embs_grad", []).append(round(sum(embs_grad) / len(embs_grad), 5))
                grad_list.setdefault("transformer_grad", []).append(
                    round(sum(transformer_grad) / len(transformer_grad), 5))
            last_training_time = time.time_ns()
        batch_id += 1

    if train_costs:
        logging.info(
            f"rank {rank} epoch {epoch}; mean cost per step: "
            f"{mean(train_costs) / batch_group / 1000000:.2f}ms"
        )
    else:
        logging.info(f"rank {rank} epoch {epoch}; less than {batch_group} steps")

    epoch_loss = avg_eval(torch.tensor([loss]).to(device), world_size=world_size)

    # 训练完成，保存模型
    model_save_pth = os.path.join(save_dir, export_conf["save_dir_name"])
    if rank == 0:
        if os.path.exists(model_save_pth):
            shutil.rmtree(model_save_pth)
        os.makedirs(model_save_pth, exist_ok=True)
    model_path = os.path.join(model_save_pth, "model_longer.pth")
    if isinstance(executor, LocalExecutor):
        # 新版代码只支持保存state_dict
        if rank == 0:
            torch.save(executor.model.module.state_dict(), model_path)
    else:
        # torchrec保存分布式模型，用于后续执行Eval
        dist_model_path = os.path.join(model_save_pth, f"model_longer_{rank}.pth")
        torch.save(executor.model.state_dict(), dist_model_path)
        # 合并分布式模型权重并保存
        gather_state_dict = executor.gather_state_dict()
        if rank == 0:
            torch.save(gather_state_dict, model_path)
    logging.info("Saving model to %s", model_path)
    return batch_id, epoch_loss


def write_to_file(save_file, mode='w', encoding=None):
    _flags = os.O_WRONLY | os.O_CREAT
    _stats = stat.S_IWUSR | stat.S_IRUSR
    if encoding is not None:
        file_hander = os.fdopen(os.open(save_file, _flags, _stats), mode, encoding=encoding)
    else:
        file_hander = os.fdopen(os.open(save_file, _flags, _stats), mode)
    return file_hander


def save_input_output(executor: Executor,
                      save_dir,
                      model_input_export,
                      export_conf,
                      feature_conf,
                      rank):
    """
    保存模型
    """

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    export_batch_size = export_conf.get('export_batch_size', 1)
    save_multi_pth_input = export_conf.get('save_multi_pth_input', False)
    exclude_features_export = export_conf.get("exclude_features_export", [])
    logging.info("exclude features when exporting %s", exclude_features_export)
    logging.info("export_batch_size is %s", export_batch_size)
    model_input = dict()
    device = model_input_export[feature_conf["infer_label_key"]].device

    for k, v in model_input_export.items():
        if k in exclude_features_export:
            continue
        v = v[:export_batch_size]
        # # 将256的序列填满
        if k in feature_conf["item_feature_columns"]:
            feature_count = feature_conf["item_feature_columns"][k]["feature_count"]
            if feature_conf["item_feature_columns"][k]["dtype"] == "int":
                num_rerank = v.size(-1)
                v[0, 0, :] = torch.randint(1, feature_count, (num_rerank,), device=device)
            elif feature_conf["item_feature_columns"][k]["dtype"] == "con":
                num_rerank = v.size(-1)
                v[0, 0, :] = torch.rand((num_rerank,), device=device)
            elif feature_conf["item_feature_columns"][k]["dtype"] == "multi":
                num_rerank = v.size(1)
                num_value = v.size(-1)
                v[0, :, :] = torch.randint(1, feature_count, (num_rerank, num_value), device=device)

        model_input[k] = v
    with torch.no_grad():
        output_1 = executor.evaluate_once(model_input)["rerank_score"]
    output = to_numpy(output_1)

    with torch.no_grad():
        output_2 = executor.evaluate_once(model_input)["rerank_score"]
    diff = output_1 - output_2
    logging.info("The difference between two runs is %s", torch.sum(diff).tolist())
    if rank == 0:
        model_input_export = dict()
        for k, v in model_input.items():
            if k in exclude_features_export or not isinstance(v, torch.Tensor):
                continue
            v = to_numpy(v)
            model_input_export[k] = v
        if save_multi_pth_input:
            user_input_save_file = "%s/%s/user_pth_input" % (save_dir, export_conf["save_dir_name"])
            if os.path.exists(user_input_save_file):
                os.remove(user_input_save_file)
            candidate_input_save_file = "%s/%s/candidate_pth_input" % (save_dir, export_conf["save_dir_name"])
            if os.path.exists(candidate_input_save_file):
                os.remove(candidate_input_save_file)
            action_input_save_file = "%s/%s/action_pth_input" % (save_dir, export_conf["save_dir_name"])
            if os.path.exists(action_input_save_file):
                os.remove(action_input_save_file)
            user_model_input_export, candidate_model_input_export, action_model_input_export = {}, {}, {}
            for k, v in model_input_export:
                if k in feature_conf["item_feature_columns"]:
                    candidate_model_input_export[k] = v
                elif k in feature_conf["user_feature_columns"]:
                    user_model_input_export[k] = v
                elif "seq" in k:
                    action_model_input_export[k] = v
            input_pickle_file = write_to_file(user_input_save_file, 'wb')
            pickle.dump(user_model_input_export, input_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
            input_pickle_file = write_to_file(candidate_input_save_file, 'wb')
            pickle.dump(candidate_model_input_export, input_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
            input_pickle_file = write_to_file(action_input_save_file, 'wb')
            pickle.dump(action_model_input_export, input_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            input_save_file = "%s/%s/pth_input" % (save_dir, export_conf["save_dir_name"])
            if os.path.exists(input_save_file):
                os.remove(input_save_file)
            input_pickle_file = write_to_file(input_save_file, 'wb')
            pickle.dump(model_input_export, input_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    
        output_save_file = "%s/%s/pth_output" % (save_dir, export_conf["save_dir_name"])
        if os.path.exists(output_save_file):
            os.remove(output_save_file)
        output_pickle_file = write_to_file(output_save_file, 'wb')
        pickle.dump(output, output_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


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

    save_feature_map_file = "%s/%s.config/feature_map.json" % (save_dir, export_conf["save_dir_name"])
    if os.path.exists(save_feature_map_file):
        os.remove(save_feature_map_file)
    fp = write_to_file(save_feature_map_file, 'w', encoding='utf-8')
    json.dump(feature_dict, fp, indent=4, ensure_ascii=False)
    logging.info("Saved feature map to %s/%s.config/feature_map.json", save_dir, export_conf["save_dir_name"])


def save_config(export_conf, save_dir, config_to_save, name="gr_module_config.json"):
    config_to_save = {"gr_module_config": config_to_save}

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists("%s/%s" % (save_dir, export_conf["save_dir_name"])):
        os.mkdir("%s/%s" % (save_dir, export_conf["save_dir_name"]))

    save_config_file = "%s/%s" % (save_dir, name)
    if os.path.exists(save_config_file):
        os.remove(save_config_file)
    fp = write_to_file(save_config_file, 'w', encoding='utf-8')
    json.dump(config_to_save, fp, indent=4, ensure_ascii=False)
    logging.info("Saved config file to %s/%s", save_dir, name)


def evaluate_step(executor: Executor,
                  feature_conf,
                  feature_map,
                  train_conf,
                  save_dir,
                  export_conf,
                  world_size,
                  rank,
                  save_after_eval):
    """
    评估模型
    """

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def merge_model_input(model_input_1, model_input_2):
        for k, v in model_input_1.items():
            v2 = model_input_2[k]
            v2 = to_numpy(v2)
            v = np.concatenate((v, v2), axis=0)
            model_input_1[k] = v
        return model_input_1


    batch_recorded = 0

    model_input_auc = None
    logging.info("rank %s starting evaluation...", rank)
    scores_all, ground_truth_all, eval_weights_all = [], [], []

    eval_iter = 0
    last_eval_time = time.perf_counter()

    is_input_copied = False

    while True:
        try:
            with torch.no_grad():
                scores_dict, model_input = executor.execute(eval_iter)
        except StopIteration:
            logging.info("finished")
            break
        ground_truth = model_input.get("label")
        B = model_input["user_id"].size(0)
        scores = scores_dict["rerank_score"]
        valid_items = model_input.get("valid_items").reshape(B, 1)
        scores = scores.reshape(B, -1)
        ground_truth = ground_truth.reshape(B, -1)
        scores = torch.cat([scores[i, :valid_items[i].item()] for i in range(scores.size(0))], dim=0)
        ground_truth = torch.cat([ground_truth[i, :valid_items[i].item()] for i in range(ground_truth.size(0))],
                                 dim=0)

        scores_all.extend(scores.view(-1).detach().cpu().tolist())
        ground_truth_all.extend(ground_truth.view(-1).detach().cpu().tolist())

        if (eval_iter % train_conf["eval_interval"]) == 0:
            torch.distributed.barrier()  # sync time taken
            cost = time.perf_counter() - last_eval_time
            eval_time = f"{cost:.2f}s"
            logging.info(
                "rank %s; batch-stat (eval): step %s (EVAL in %s)", rank, eval_iter, eval_time)
            last_eval_time = time.perf_counter()

        if not is_input_copied:
            model_input_export = copy.deepcopy(model_input)
            is_input_copied = True
            
        eval_iter += 1

    torch_npu.npu.empty_cache()
    logging.info("rank %s start gather...", rank)
    ground_truth_all = gather_all_list(ground_truth_all, world_size=world_size)
    scores_all = gather_all_list(scores_all, world_size=world_size)

    logging.info(
        'Number of samples in eval dataset: %s, rank %s, number of positive samples %s, '
        'number of negative samples %s',
        len(scores_all), rank, sum(ground_truth_all), len(ground_truth_all) - sum(ground_truth_all))

    if not is_input_copied:
        logging.error("No input sample can be saved because all the candidates are zero.")
    save_result_to_local = feature_conf.get("save_result_to_local", True)
    if rank == 0:
        save_path = "%s/modelfile" % save_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        logging.info("Test data count: %d", len(ground_truth_all))
        metrics_file_name = "metrics_report.csv"
        metrics_path = os.path.join(save_path, metrics_file_name)
        if save_result_to_local:
            logging.info("Saving metrics to %s", metrics_path)
        eval_df = pd.DataFrame({
            'ground_truth': ground_truth_all,
            'scores': scores_all,
        })
                
        calculator = MetricsCalculator(eval_df, feature_map, feature_conf.get('metric_list', []),
                                       feature_conf.get('group_metric_cols_values', []),
                                       feature_conf.get('bias_evaluation_conf', []), save_result_to_local)
        calculator.calculate(metrics_path)

    if save_after_eval:
        os.makedirs(save_dir, exist_ok=True)
        save_input_output(executor, save_dir, model_input_export, export_conf, feature_conf, rank)

    logging.info(
        'Number of samples in eval dataset: %s, rank %s, number of positive samples %s, '
        'number of negative samples %s',
        len(scores_all), rank, sum(ground_truth_all), len(ground_truth_all) - sum(ground_truth_all))
    return


def train_fn(config, data_dir, save_dir, feature_map_dir_or_path, LLM_embedding_dir, period) -> None:
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
    logging.info("world size is %s, node number is %s", world_size, node_num)

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
    num_epochs = train_conf.get("num_epochs", 1)

    if dataset_name == "ag-rank":
        myclass.set_path(rootdir=save_dir, filename="analysis.csv")

    logging.info("Training model on rank %s (local rank: %s); device: %s; world_size: %s;",
                 rank, local_rank, device, world_size)

    init_torch_config(local_rank, train_conf)

    # 1.2 这一部分根据业务数据格式，需要用特定的方式编辑feat_conf和model_conf

    feature_conf, model_conf, feature_map = refine_feat_and_model_conf(dataset=dataset_name,
                                                                       feature_conf=feature_conf,
                                                                       model_conf=model_conf,
                                                                       feature_map_dir_or_path=feature_map_dir_or_path,
                                                                       model_cfg=config[Const.MODEL_CFG])
    # 1.3 保存feature_map供后续模型调用
    if rank == 0:
        save_feature_map(export_conf=export_conf,
                         save_dir=save_dir,
                         feature_map_dir_or_path=feature_map_dir_or_path,
                         feature_map=feature_map)
    # 2. 初始化数据集
    _, _, train_data_loader = get_data_loaders(dataset=dataset_name,
                                               data_dir=data_dir,
                                               rank=rank,
                                               world_size=world_size,
                                               train_config=train_conf,
                                               model_conf=model_conf,
                                               feature_config=feature_conf,
                                               dataloader_config=data_loader_conf)
    # 3. 初始化模型
    common_config['feature_conf'] = feature_conf
    model_conf["data_dir"] = data_dir
    model_conf["save_dir"] = save_dir
    model_conf["feature_map_dir_or_path"] = feature_map_dir_or_path
    model_conf["LLM_embedding_dir"] = LLM_embedding_dir
    ModelRegistry.register_all_modules(
        modul_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "modeling/generic/sequential")))
    model = ModelInitializer.init(gr_module_cfg=config)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Total number of parameters: {total_params:.2f}M")
    total_params = sum(p.numel() for x, p in model.named_parameters() if '_emb' not in x) / 1e6
    logging.info(f"The number of parameters (exl. emb): {total_params:.2f}M")

    """
    如果模型中确实存在未使用的参数（例如某些分支仅在特定条件下执行），可以通过设置 find_unused_parameters=True 来解决
    注意：启用此选项可能会略微降低性能
    """
    if dataset_name == "ag-rank":
        train_conf["find_unused_parameters"] = True
    else:
        train_conf["find_unused_parameters"] = False

    if model_conf.get("root_model_type") == "LongerEp":
        from modeling.generic.executors.torchrec_executor import TorchrecExecutor
        executor = TorchrecExecutor(model, train_conf, world_size, node_num, device)
    else:
        executor = LocalExecutor(model, train_conf, local_rank, device)

    batch_id = 0
    for epoch in range(num_epochs):
        executor.train(train_data_loader)
        train_step_para = {
            "executor": executor,
            "train_conf": train_conf,
            "export_conf": export_conf,
            "batch_id": batch_id,
            "world_size": world_size,
            "rank": rank,
            "epoch": epoch,
            "device": device,
            "save_dir": save_dir,
            "local_rank": local_rank
        }

        batch_id, epoch_loss = train_step(**train_step_para)

        logging.info("loss at epoch %s is %s", epoch, epoch_loss)

    if dataset_name == "ag-rank":
        # 保存统计结果
        myclass.save_data(rank)

    del executor
    del model


def eval_fn(config, data_dir, save_dir, feature_map_dir_or_path, LLM_embedding_dir, period, already_init=False) -> None:
    """
    单独评估函数
    """
    # 1. 初始化随机数、加速器，配置文件
    # 1.1 这一部分可以通用
    common_config = config[Const.COMMON_HP]
    init_random_seed(common_config)
    device, rank, local_rank, world_size, node_num = init_ddp_info_params(already_init)
    logging.info("world size is %s, node number is %s", world_size, node_num)

    train_conf = common_config['train_conf']
    feature_conf = common_config['feature_conf']
    export_conf = common_config['export_conf']
    data_loader_conf = common_config['data_loader_conf']
    model_conf = common_config['model_conf']
    dataset_name = data_loader_conf["dataset_name"]
    feature_conf["period"] = period
    if dataset_name == "ag-rank":
        myclass.set_path(rootdir=save_dir, filename="analysis.csv")

    logging.info("Evaluating model on rank %s (local rank: %s); device: %s; world_size: %s;",
                 rank, local_rank, device, world_size)

    init_torch_config(local_rank, train_conf)

    # 1.2 这一部分根据业务数据格式，需要用特定的方式编辑feat_conf和model_conf

    feature_conf, model_conf, feature_map = refine_feat_and_model_conf(dataset=dataset_name,
                                                                       feature_conf=feature_conf,
                                                                       model_conf=model_conf,
                                                                       feature_map_dir_or_path=feature_map_dir_or_path,
                                                                       model_cfg=config[Const.MODEL_CFG])
    # 1.3 将配置文件输出
    common_config['feature_conf'] = feature_conf
    common_config['model_conf'] = model_conf
    config["common_hp"] = common_config
    model_conf["data_dir"] = data_dir
    model_conf["save_dir"] = save_dir
    model_conf["feature_map_dir_or_path"] = feature_map_dir_or_path
    model_conf["LLM_embedding_dir"] = LLM_embedding_dir
    if rank == 0:
        save_config(export_conf, save_dir, config, name="gr_module_config.json")

    # 2. 初始化数据集
    _, eval_data_loader, _ = get_data_loaders(dataset=dataset_name,
                                              data_dir=data_dir,
                                              rank=rank,
                                              world_size=world_size,
                                              train_config=train_conf,
                                              model_conf=model_conf,
                                              feature_config=feature_conf,
                                              dataloader_config=data_loader_conf)

    # 6.初始化模型
    ModelRegistry.register_all_modules(
        modul_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "modeling/generic/sequential")))
    model = ModelInitializer.init(gr_module_cfg=config)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Total number of parameters: {total_params:.2f}M")
    total_params = sum(p.numel() for x, p in model.named_parameters() if '_emb' not in x) / 1e6
    logging.info(f"The number of parameters (exl. emb): {total_params:.2f}M")

    """
    如果模型中确实存在未使用的参数（例如某些分支仅在特定条件下执行），可以通过设置 find_unused_parameters=True 来解决
    注意：启用此选项可能会略微降低性能
    """
    if dataset_name == "ag-rank":
        train_conf["find_unused_parameters"] = True
    else:
        train_conf["find_unused_parameters"] = False


    if model_conf.get("root_model_type") == "LongerEp":
        from modeling.generic.executors.torchrec_executor import TorchrecExecutor
        # 当采用load_state_dict加载时，跳过跨机场景worldsize校验
        shardedTensor_patched_setstate()
        executor = TorchrecExecutor(model, train_conf, world_size, node_num, device)
        # 使用torchrec训练save的模型是shard过的，加载时需要单独加载
        model_path = os.path.join(save_dir, export_conf["save_dir_name"], f"model_longer_{rank}.pth")
        executor.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    else:
        # 加载模型
        model_path = os.path.join(save_dir, export_conf["save_dir_name"], "model_longer.pth")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        executor = LocalExecutor(model, train_conf, local_rank, device)
    executor.eval(eval_data_loader)

    if dataset_name == "ag-rank":
        # 保存统计结果
        myclass.save_data(rank)
    # 6. 测试模型
    evaluate_step(executor=executor,
                  feature_conf=feature_conf,
                  feature_map=feature_map,
                  train_conf=train_conf,
                  save_dir=save_dir,
                  export_conf=export_conf,
                  world_size=world_size,
                  rank=rank,
                  save_after_eval=True)


def shardedTensor_patched_setstate():
    def patched_setstate(self, state):
        self._sharded_tensor_id = None
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Need to initialize default process group using "
                '"init_process_group" before loading ShardedTensor'
            )

        (
            self._local_shards,
            self._metadata,
            pg_state,
            self._sharding_spec,
            self._init_rrefs,
        ) = state

        # 跳过 pg 校验：你可以打印一下值确认
        from torch.distributed._shard.api import _get_current_process_group
        self._process_group = _get_current_process_group()

        self._post_init()

    # monkey patch
    ShardedTensor.__setstate__ = patched_setstate


def get_train_eval_dataloader(dataset, data_dir, local_batch_size, eval_batch_size, max_sequence_length, rank,
                              world_size,
                              feature_config, dataloader_config, data_pth=''):
    prefetch_factor = dataloader_config['prefetch_factor']
    num_workers_train = dataloader_config['num_workers_train']
    num_workers_val = dataloader_config['num_workers_val']

    dataset = get_reco_dataset(
        dataset=dataset,
        data_dir=data_dir,
        pth=data_pth,
        rank=rank,
        world_size=world_size,
        max_sequence_length=max_sequence_length,
        chronological=True,
        feature_conf=feature_config,
        num_rerank=dataloader_config.get("num_rerank", 256)
    )
    train_data_loader = create_data_loader(
        dataset.train_dataset,
        batch_size=local_batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers_train
    )
    eval_data_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=eval_batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers_val
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
                 FLAGS.LLM_embedding_dir,
                 str(FLAGS.period))
        torch_npu.npu.empty_cache()
        eval_fn(config,
                FLAGS.data_dir,
                FLAGS.save_dir,
                FLAGS.feature_map_dir,
                FLAGS.LLM_embedding_dir,
                str(FLAGS.period),
                True)
    else:
        eval_fn(config,
                FLAGS.data_dir,
                FLAGS.save_dir,
                FLAGS.feature_map_dir,
                FLAGS.LLM_embedding_dir,
                str(FLAGS.period))


if __name__ == "__main__":
    app.run(main_torchrun)
