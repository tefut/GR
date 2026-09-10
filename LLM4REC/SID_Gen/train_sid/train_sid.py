# -*- coding: utf-8 -*-
"""
RQVAE 训练脚本
支持从 YAML 配置加载参数

配置优先级（从低到高）：
1. configs/default.yaml - 默认配置
2. configs/game.yaml（通过 --config 指定）- 业务配置
3. CLI 参数 - 命令行覆盖
"""

import argparse
import logging
import os
import re
from collections import OrderedDict

import numpy as np
import torch
import torch_npu
import wandb
from SID_Gen.models.faiss_rq import (
    load_pretrained_codebook,
    train_faiss_rq,
    encode_with_rq,
    sinkhorn_uniform_mapping,
    save_codebook_npz,
    save_faiss_index,
)
from SID_Gen.models.rqkmeans_constrained import (
    HAS_CONSTRAINED,
    residual_kmeans_constrained,
)
from SID_Gen.models.rqkmeans_plus import (
    ResidualEncoderWrapper,
    apply_rqkmeans_plus_strategy,
)
from SID_Gen.models.rqvae import RQVAE
from SID_Gen.my_datasets.emb_datasets import EmbDataset, create_emb_dataset
from SID_Gen.train_sid.trainer import Trainer
from torch_npu.contrib import transfer_to_npu

FAISS_AVAILABLE = True

# 使用统一的配置加载器
from SID_Gen.utils.config_loader import load_yaml_config
from SID_Gen.utils.log_utils import get_logger
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from torch.utils.data import DataLoader

# ----------------------------
# 常量定义
# ----------------------------
SUPPORTED_MODEL_TYPES = ["rqvae", "rqkmeans", "rqkmeans_plus", "rqkmeans_constrained"]


# ----------------------------
# 辅助函数
# ----------------------------
def _log_namespace(logger: logging.Logger, args):
    """统一用 %s 打印所有参数"""
    logger.info("=================================================")
    for k in sorted(args.__dict__.keys()):
        logger.info("%s = %s", k, getattr(args, k))
    logger.info("=================================================")


# ----------------------------
# Checkpoint 加载工具
# ----------------------------
def load_ckpt_cpu(path: str):
    """兼容 torch/torch_npu 不同版本的 torch.load"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(ckpt):
    """
    从各种常见保存格式里提取 state_dict:
    - torch.save(model.state_dict())
    - torch.save({"model": sd, ...})
    - torch.save({"state_dict": sd, ...})
    - torch.save({"model_state_dict": sd, ...})
    """
    if isinstance(ckpt, OrderedDict):
        return ckpt

    if isinstance(ckpt, dict):
        for k in ["model", "model_state_dict", "state_dict", "net", "module", "params"]:
            v = ckpt.get(k, None)
            if isinstance(v, (dict, OrderedDict)) and len(v) > 0:
                return v

        if any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt

    raise ValueError(
        "Cannot extract state_dict from ckpt. type=%s, keys=%s"
        % (type(ckpt), list(ckpt.keys()) if isinstance(ckpt, dict) else None)
    )


def strip_prefix_if_needed(state_dict):
    """
    自动处理 key 前缀不一致：
    - module.xxx
    - model.xxx
    - model.module.xxx
    - net.xxx
    以及截断到 encoder./decoder./rq.
    """
    keys = list(state_dict.keys())
    if not keys:
        return state_dict

    if any(keys[0].startswith(p) for p in ("encoder.", "decoder.", "rq.")):
        return state_dict

    for pref in ("module.", "model.", "model.module.", "net."):
        head_n = min(100, len(keys))
        if all(k.startswith(pref) for k in keys[:head_n]):
            return OrderedDict((k[len(pref):], v) for k, v in state_dict.items())

    for anchor in ("encoder.", "decoder.", "rq."):
        if any(anchor in k for k in keys):
            def cut(k, anchor=anchor):
                i = k.find(anchor)
                return k[i:] if i >= 0 else k

            return OrderedDict((cut(k), v) for k, v in state_dict.items())

    return state_dict


_COLLISION_RE = re.compile(r"collision_(\d+(?:\.\d+)?)\.pt$", re.IGNORECASE)


def _parse_collision_from_name(fname: str):
    """
    从文件名中解析 collision 数值：
    e.g. best_collision_epoch_000001_collision_0.459848.pt -> 0.459848
    """
    m = _COLLISION_RE.search(fname)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _select_ckpt_from_dir(
        ckpt_dir: str,
        logger: logging.Logger,
        prefer_prefixes=("best_collision_", "best_loss_", "epoch_epoch_"),
):
    """
    在目录中挑 ckpt：
    1) 按 prefer_prefixes 的优先级分组
    2) 每组里按 collision 最小选择
    """
    if not os.path.isdir(ckpt_dir):
        raise NotADirectoryError(f"Not a directory: {ckpt_dir}")

    files = []
    for fn in os.listdir(ckpt_dir):
        if fn.endswith(".pt"):
            col = _parse_collision_from_name(fn)
            if col is not None:
                files.append((fn, col))

    if not files:
        raise FileNotFoundError(
            f"No .pt files with pattern '*collision_XXX.pt' found in: {ckpt_dir}"
        )

    # 分组选择：按前缀优先级
    for prefix in prefer_prefixes:
        candidates = [(fn, col) for (fn, col) in files if fn.startswith(prefix)]
        if candidates:
            best_fn, best_col = min(candidates, key=lambda x: x[1])
            best_path = os.path.join(ckpt_dir, best_fn)
            logger.info(
                "[Init] Auto-selected ckpt by prefix='%s': %s (collision=%.6f)",
                prefix, best_path, best_col
            )
            return best_path, best_col

    # 如果没有任何前缀匹配，就全量取 collision 最小
    best_fn, best_col = min(files, key=lambda x: x[1])
    best_path = os.path.join(ckpt_dir, best_fn)
    logger.info(
        "[Init] Auto-selected ckpt (no preferred prefix matched): %s (collision=%.6f)",
        best_path, best_col
    )
    return best_path, best_col


def load_pretrained_if_needed(
        args,
        model,
        accelerator: Accelerator,
        logger: logging.Logger,
):
    """
    args.pretrained_ckpt:
      - None/""：不加载
      - 具体文件路径：按原逻辑加载
      - 目录路径：自动选 collision 最小的 best_collision_*（找不到则降级）
    """
    if not getattr(args, "pretrained_ckpt", None):
        return

    ckpt_input = args.pretrained_ckpt

    # 如果传的是目录：自动选最优 ckpt
    if os.path.isdir(ckpt_input):
        if accelerator.is_main_process:
            try:
                selected_path, selected_col = _select_ckpt_from_dir(
                    ckpt_input, logger=logger
                )
            except Exception as e:
                logger.exception("[Init] Failed to auto-select ckpt from dir: %s", ckpt_input)
                raise
            ckpt_path = selected_path
        else:
            ckpt_path = None

        # 广播 ckpt_path 给所有进程
        obj_list = [ckpt_path]
        broadcast_object_list(obj_list)
        ckpt_path = obj_list[0]
    else:
        ckpt_path = ckpt_input

    # 把 args.pretrained_ckpt 换成 ckpt_path
    if accelerator.is_main_process:
        logger.info("[Init] Loading pretrained checkpoint from: %s", ckpt_path)
        ckpt = load_ckpt_cpu(ckpt_path)
        sd = extract_state_dict(ckpt)
        sd = strip_prefix_if_needed(sd)
        logger.info("[Init] Example ckpt keys: %s", list(sd.keys())[:10])
    else:
        sd = None

    obj_list = [sd]
    broadcast_object_list(obj_list)
    sd = obj_list[0]

    missing, unexpected = model.load_state_dict(sd, strict=False)

    if accelerator.is_main_process:
        total = len(model.state_dict())
        loaded = total - len(missing)
        logger.info("[Init] Loaded params: %s/%s (%.2f%%)", loaded, total, loaded / max(total, 1) * 100.0)

        for k in [
            "encoder.mlp_layers.0.weight",
            "rq.vq_layers.0.embedding.weight",
            "decoder.mlp_layers.0.weight",
        ]:
            if k in model.state_dict():
                logger.info("[Init] Param norm %s: %.6f", k, model.state_dict()[k].norm().item())

        if missing and len(missing) <= 80:
            logger.info("[Init] Missing keys: %s", missing)
        if unexpected and len(unexpected) <= 80:
            logger.info("[Init] Unexpected keys: %s", unexpected)


# ----------------------------
# 配置加载和合并
# ----------------------------
def load_config(config_path: str, default_config_path: str = "configs/default.yaml") -> dict:
    """加载 YAML 配置（支持默认配置 + 业务配置的合并）"""
    return load_yaml_config(config_path, default_config_path)


def _merge_config_with_args(config: dict, args) -> argparse.Namespace:
    """
    将 YAML 配置与 CLI 参数合并，CLI 参数优先

    优先级（从低到高）：
    1. 默认配置 (configs/default.yaml)
    2. 业务配置 (configs/game.yaml)
    3. CLI 参数
    """
    train_cfg = config.get("train_sid", {})

    merged = argparse.Namespace()

    # 数据和路径（优先使用 CLI 参数）
    merged.data_type = _get_value(args.data_type, train_cfg, "data_type", "npz")
    merged.data_path = _get_value(args.data_path, train_cfg, "data_path", "")
    merged.data_dir = _get_value(args.data_dir, train_cfg, "data_dir", "")
    merged.manifest_path = _get_value(args.manifest_path, train_cfg, "manifest_path", "")
    merged.columns = _get_value(args.columns, train_cfg, "columns", None)
    merged.ckpt_dir = _get_value(args.ckpt_dir, train_cfg, "ckpt_dir", "")
    merged.log_file = _get_value(args.log_file, train_cfg, "log_file", "")

    # 流式加载配置
    merged.max_cache_shards = _get_value(args.max_cache_shards, train_cfg, "max_cache_shards", 4)
    merged.shuffle_shards = _get_value(args.shuffle_shards, train_cfg, "shuffle_shards", False)
    merged.prefetch_shards = _get_value(args.prefetch_shards, train_cfg, "prefetch_shards", 0)

    # 训练参数
    merged.lr = _get_value(args.lr, train_cfg, "lr", 5e-4)
    merged.epochs = _get_value(args.epochs, train_cfg, "epochs", 2000)
    merged.batch_size = _get_value(args.batch_size, train_cfg, "batch_size", 1024)
    merged.num_workers = _get_value(args.num_workers, train_cfg, "num_workers", 4)
    merged.eval_step = _get_value(args.eval_step, train_cfg, "eval_step", 5)
    merged.learner = _get_value(args.learner, train_cfg, "learner", "AdamW")
    merged.lr_scheduler_type = _get_value(args.lr_scheduler_type, train_cfg, "lr_scheduler_type", "constant")
    merged.warmup_epochs = _get_value(args.warmup_epochs, train_cfg, "warmup_epochs", 50)
    merged.weight_decay = _get_value(args.weight_decay, train_cfg, "weight_decay", 0.0)

    # 模型参数
    merged.dropout_prob = _get_value(args.dropout_prob, train_cfg, "dropout_prob", 0.0)
    merged.bn = _get_value(args.bn, train_cfg, "bn", False)
    merged.loss_type = _get_value(args.loss_type, train_cfg, "loss_type", "mse")
    merged.kmeans_init = _get_value(args.kmeans_init, train_cfg, "kmeans_init", True)
    merged.kmeans_iters = _get_value(args.kmeans_iters, train_cfg, "kmeans_iters", 100)
    merged.sk_epsilons = _get_value(args.sk_epsilons, train_cfg, "sk_epsilons", [0.0, 0.0, 0.0])
    merged.sk_iters = _get_value(args.sk_iters, train_cfg, "sk_iters", 50)
    merged.num_emb_list = _get_value(args.num_emb_list, train_cfg, "num_emb_list", [256, 256, 256])
    merged.e_dim = _get_value(args.e_dim, train_cfg, "e_dim", 32)
    merged.quant_loss_weight = _get_value(args.quant_loss_weight, train_cfg, "quant_loss_weight", 1.0)
    merged.recon_weight = _get_value(args.recon_weight, train_cfg, "recon_weight", 1.0)
    merged.beta = _get_value(args.beta, train_cfg, "beta", 0.25)
    merged.layers = _get_value(args.layers, train_cfg, "layers", [2048, 1024, 512, 256, 128, 64])

    # 设备
    merged.device = config.get("device", train_cfg.get("device", "npu"))

    # 预训练和其他
    merged.pretrained_ckpt = _get_value(args.pretrained_ckpt, train_cfg, "pretrained_ckpt", "")
    merged.pretrained_codebook_path = _get_value(
        args.pretrained_codebook_path, train_cfg, "pretrained_codebook_path", ""
    )
    merged.model_type = _get_value(args.model_type, train_cfg, "model_type", "rqvae")
    merged.save_limit = _get_value(args.save_limit, train_cfg, "save_limit", 5)
    merged.eval_dump_root = _get_value(args.eval_dump_root, train_cfg, "eval_dump_root", "../../data")
    merged.dump_sids = _get_value(args.dump_sids, train_cfg, "dump_sids", True)
    merged.dump_sids_format = _get_value(args.dump_sids_format, train_cfg, "dump_sids_format", "json")

    # 日志和随机种子
    merged.seed = _get_value(args.seed, train_cfg, "seed", 2024)
    merged.log_level = _get_value(args.log_level, train_cfg, "log_level", "INFO")

    # 数据预处理
    merged.norm = _get_value(args.norm, train_cfg, "norm", False)
    merged.csv_sep = _get_value(args.csv_sep, train_cfg, "csv_sep", ",")
    merged.expected_emb_dim = _get_value(
        args.expected_emb_dim, train_cfg, "expected_emb_dim", None
    )

    # 列名映射配置
    merged.column_mapper = config.get("column_mapping", {})

    # WandB 配置
    merged.use_wandb = _get_value(args.use_wandb, train_cfg, "use_wandb", False)
    merged.wandb_project = _get_value(args.wandb_project, train_cfg, "wandb_project", "SID_Gen")
    merged.wandb_entity = _get_value(args.wandb_entity, train_cfg, "wandb_entity", None)
    merged.wandb_tags = _get_value(args.wandb_tags, train_cfg, "wandb_tags", [])

    # Dead code reset 配置
    merged.enable_dead_code_reset = _get_value(args.enable_dead_code_reset, train_cfg, "enable_dead_code_reset", False)
    merged.reset_threshold = _get_value(args.reset_threshold, train_cfg, "reset_threshold", 1.0)
    merged.reset_freq = _get_value(args.reset_freq, train_cfg, "reset_freq", 100)
    merged.ema_decay = _get_value(args.ema_decay, train_cfg, "ema_decay", 0.99)

    # RQKMeans 专用配置
    merged.uniform_mapping = _get_value(args.uniform_mapping, train_cfg, "uniform_mapping", False)
    merged.uniform_iters = _get_value(args.uniform_iters, train_cfg, "uniform_iters", 30)
    merged.uniform_tau = _get_value(args.uniform_tau, train_cfg, "uniform_tau", None)

    return merged


def _get_value(cli_value, cfg: dict, key: str, default):
    """
    获取配置值，CLI 参数优先

    Args:
        cli_value: 命令行参数值
        cfg: YAML 配置字典
        key: 配置键名
        default: 默认值

    Returns:
        最终使用的值
    """
    # CLI 参数存在且不是 None 时优先使用
    if cli_value is not None:
        return cli_value

    # 否则使用配置文件中的值
    return cfg.get(key, default)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RQVAE 训练脚本')

    # === 配置文件 ===
    parser.add_argument(
        '--config',
        type=str,
        default='configs/game.yaml',
        help='配置文件路径（YAML格式）'
    )

    # === 数据和路径 ===
    parser.add_argument('--data_type', type=str, default=None, help='数据类型: npz 或 parquet')
    parser.add_argument('--data_path', type=str, default=None, help='npz数据文件路径')
    parser.add_argument('--data_dir', type=str, default=None, help='parquet数据目录')
    parser.add_argument('--manifest_path', type=str, default=None, help='Manifest文件路径（用于parquet）')
    parser.add_argument('--columns', type=str, nargs='+', default=None, help='需要读取的列名列表')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='checkpoint 保存目录')
    parser.add_argument('--log_file', type=str, default=None, help='日志文件路径')

    # === 流式加载配置 ===
    parser.add_argument('--max_cache_shards', type=int, default=None, help='最大缓存分片数')
    parser.add_argument('--shuffle_shards', action='store_true', default=None, help='打乱分片顺序')
    parser.add_argument('--no-shuffle_shards', dest='shuffle_shards', action='store_false', help='不打乱分片顺序')
    parser.add_argument('--prefetch_shards', type=int, default=None, help='预取分片数')

    # === 训练参数 ===
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小')
    parser.add_argument('--num_workers', type=int, default=None, help='DataLoader workers 数量')
    parser.add_argument('--eval_step', type=int, default=None, help='评估间隔（epoch）')
    parser.add_argument('--learner', type=str, default=None, help='优化器类型')
    parser.add_argument('--lr_scheduler_type', type=str, default=None, help='学习率调度器类型')
    parser.add_argument('--warmup_epochs', type=int, default=None, help='预热轮数')
    parser.add_argument('--weight_decay', type=float, default=None, help='权重衰减')

    # === 模型参数 ===
    parser.add_argument('--dropout_prob', type=float, default=None, help='Dropout 概率')
    parser.add_argument('--bn', action='store_true', help='是否使用 BatchNorm')
    parser.add_argument('--no-bn', dest='bn', action='store_false', help='不使用 BatchNorm')
    parser.add_argument('--loss_type', type=str, default=None, help='损失函数类型')
    parser.add_argument('--kmeans_init', action='store_true', default=None, help='使用 KMeans 初始化')
    parser.add_argument('--no-kmeans_init', dest='kmeans_init', action='store_false', help='不使用 KMeans 初始化')
    parser.add_argument('--kmeans_iters', type=int, default=None, help='KMeans 迭代次数')
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=None, help='Sinkhorn epsilon 值')
    parser.add_argument('--sk_iters', type=int, default=None, help='Sinkhorn 迭代次数')
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=None, help='各层 embedding 数量')
    parser.add_argument('--e_dim', type=int, default=None, help='Embedding 维度')
    parser.add_argument('--quant_loss_weight', type=float, default=None, help='量化损失权重')
    parser.add_argument('--recon_weight', type=float, default=None, help='重构损失权重')
    parser.add_argument('--beta', type=float, default=None, help='Beta 参数')
    parser.add_argument('--layers', type=int, nargs='+', default=None, help='网络层维度列表')

    # === 预训练 ===
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='预训练 checkpoint 路径')
    parser.add_argument('--pretrained_codebook_path', type=str, default=None,
                        help='预训练 codebook 路径（rqkmeans 模式专用）')
    parser.add_argument('--model_type', type=str, default=None,
                        help='模型类型: rqvae / rqkmeans / rqkmeans_plus / rqkmeans_constrained')

    # === 评估输出 ===
    parser.add_argument('--save_limit', type=int, default=None, help='最多保存的 checkpoint 数量')
    parser.add_argument('--eval_dump_root', type=str, default=None, help='评估结果输出目录')
    parser.add_argument('--dump_sids', action='store_true', default=None, help='导出 SID')
    parser.add_argument('--no-dump_sids', dest='dump_sids', action='store_false', help='不导出 SID')
    parser.add_argument('--dump_sids_format', type=str, default=None, help='SID 导出格式')

    # === 其他 ===
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--log_level', type=str, default=None, help='日志级别')
    parser.add_argument('--norm', action='store_true', default=None, help='是否对输入embedding做L2归一化')
    parser.add_argument('--no-norm', dest='norm', action='store_false', help='不对输入embedding做L2归一化')
    parser.add_argument('--csv_sep', type=str, default=None, help='CSV文件分隔符（用于data_type=csv）')
    parser.add_argument('--expected_emb_dim', type=int, default=None,
                        help='期望的embedding维度，丢弃维度不匹配的脏数据')

    # === WandB ===
    parser.add_argument('--use_wandb', action='store_true', default=None, help='是否启用wandb日志')
    parser.add_argument('--no-use_wandb', dest='use_wandb', action='store_false', help='禁用wandb日志')
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB项目名称')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB实体/团队名称')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None, help='WandB标签列表')

    # === Dead Code Reset ===
    parser.add_argument('--enable_dead_code_reset', action='store_true', default=None, help='启用死码重置功能')
    parser.add_argument('--no-enable_dead_code_reset', dest='enable_dead_code_reset', \
                        action='store_false', help='禁用死码重置功能')
    parser.add_argument('--reset_threshold', type=float, default=None, help='死码判定阈值（使用次数低于此值认死亡）')
    parser.add_argument('--reset_freq', type=int, default=None, help='死码重置频率（每多少步执行一次）')
    parser.add_argument('--ema_decay', type=float, default=None, help='EMA 衰减系数')

    # === RQKMeans 专用配置 ===
    parser.add_argument('--uniform_mapping', action='store_true', default=None,
                        help='启用 Sinkhorn 均匀映射（rqkmeans 模式专用）')
    parser.add_argument('--no-uniform_mapping', dest='uniform_mapping', action='store_false',
                        help='禁用 Sinkhorn 均匀映射')
    parser.add_argument('--uniform_iters', type=int, default=None,
                        help='Sinkhorn 均匀映射迭代次数')
    parser.add_argument('--uniform_tau', type=float, default=None,
                        help='Sinkhorn 正则化参数 tau')

    return parser.parse_args()


# ----------------------------
# 模型类型路由工厂函数
# ----------------------------
def create_rqvae_model(args, data, accelerator, logger):
    """
    创建标准 RQVAE 模型

    Parameters:
        args: 合并后的配置参数
        data: 数据集实例
        accelerator: Accelerator 实例
        logger: 日志记录器

    Returns:
        RQVAE(nn.Module) 对象
    """
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        recon_weight=args.recon_weight,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        enable_dead_code_reset=args.enable_dead_code_reset,
        reset_threshold=args.reset_threshold,
        reset_freq=args.reset_freq,
        ema_decay=args.ema_decay,
    )

    load_pretrained_if_needed(args, model, accelerator, logger)
    return model


def create_rqkmeans_model(args, data, accelerator, logger):
    """
    创建 FAISS 量化器（RQKMeans 模式）

    返回一个 dict 包含：
        - "rq": faiss.ResidualQuantizer 对象
        - "codebook_size": 每层 codebook 大小
        - "num_levels": 量化层数

    注意：RQKMeans 使用独立的 FAISS 训练流程，不使用 Trainer

    Parameters:
        args: 合并后的配置参数
        data: 数据集实例
        accelerator: Accelerator 实例
        logger: 日志记录器

    Returns:
        dict: 包含 FAISS 量化器的字典
    """
    import faiss

    codebook_size = args.num_emb_list[0]
    num_levels = len(args.num_emb_list)
    nbits = int(np.log2(codebook_size))

    if args.pretrained_codebook_path:
        # 加载预训练的量化器
        from SID_Gen.models.faiss_rq import load_faiss_quantizer
        logger.info(
            "[Init] Loading pretrained FAISS quantizer from: %s",
            args.pretrained_codebook_path
        )
        rq = load_faiss_quantizer(args.pretrained_codebook_path)
    else:
        # 创建新的量化器（将由 train_faiss_rq() 训练）
        logger.info(
            "[Init] Creating new FAISS ResidualQuantizer "
            "(codebook_size=%d, num_levels=%d)",
            codebook_size, num_levels
        )
        rq = faiss.ResidualQuantizer(data.dim, num_levels, nbits)

    return {
        "rq": rq,
        "codebook_size": codebook_size,
        "num_levels": num_levels,
        "dim": data.dim,
    }


def create_rqkmeans_plus_model(args, data, accelerator, logger):
    """
    创建 RQKMeans+ 模型（带残差编码器包装器）

    RQKMeans+ = RQVAE + ResidualEncoderWrapper + ZeroInit + 加载 codebook

    Parameters:
        args: 合并后的配置参数
        data: 数据集实例
        accelerator: Accelerator 实例
        logger: 日志记录器

    Returns:
        RQVAE(nn.Module) 对象（已包装）
    """
    if not args.pretrained_codebook_path:
        raise ValueError(
            "pretrained_codebook_path is required for rqkmeans_plus model type."
        )

    # 创建基础 RQVAE
    model = create_rqvae_model(args, data, accelerator, logger)

    # 应用 Plus 策略（残差连接 + 零初始化 + 加载 codebook）
    logger.info("[Init] Applying RQKMeans+ strategy")
    model = apply_rqkmeans_plus_strategy(
        model, args.pretrained_codebook_path, args.device
    )

    return model


def create_rqkmeans_constrained_model(args, data, accelerator, logger):
    """
    创建 RQKMeans Constrained 模型（使用平衡约束聚类）

    Parameters:
        args: 合并后的配置参数
        data: 数据集实例
        accelerator: Accelerator 实例
        logger: 日志记录器

    Returns:
        dict: 包含以下键的字典
            - codebooks: 多层 codebook 列表
            - codebook_size: 每层 codebook 大小
            - num_levels: 量化层数
            - dim: embedding 维度
    """
    if not HAS_CONSTRAINED:
        raise ImportError(
            "k-means-constrained library is required for rqkmeans_constrained model. "
            "Install with: pip install k-means-constrained"
        )

    codebook_size = args.num_emb_list[0]
    num_levels = len(args.num_emb_list)

    logger.info(
        "[Init] Creating RQKMeans Constrained model "
        "(codebook_size=%d, num_levels=%d)",
        codebook_size, num_levels
    )

    return {
        "codebooks": None,  # 训练后填充
        "codebook_size": codebook_size,
        "num_levels": num_levels,
        "dim": data.dim,
    }


def create_model_by_type(args, data, accelerator, logger):
    """
    根据 model_type 创建对应的模型实例

    Parameters:
        args: 合并后的配置参数
        data: 数据集实例
        accelerator: Accelerator 实例
        logger: 日志记录器

    Returns:
        - rqvae:      RQVAE(nn.Module)对象
        - rqkmeans:   dict{"rq": faiss.ResidualQuantizer, ...}
        - rqkmeans_plus: RQVAE(nn.Module)对象（已包装）

    Raises:
        ValueError: 不支持的 model_type
    """
    model_type = args.model_type

    # 验证 model_type
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Supported types: {SUPPORTED_MODEL_TYPES}"
        )

    if model_type == "rqvae":
        return create_rqvae_model(args, data, accelerator, logger)
    elif model_type == "rqkmeans":
        return create_rqkmeans_model(args, data, accelerator, logger)
    elif model_type == "rqkmeans_plus":
        return create_rqkmeans_plus_model(args, data, accelerator, logger)
    elif model_type == "rqkmeans_constrained":
        return create_rqkmeans_constrained_model(args, data, accelerator, logger)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def is_faiss_model(model):
    """
    判断模型是否为 FAISS 量化器对象

    rqkmeans 模式返回 dict，rqvae/rqkmeans_plus 返回 nn.Module

    Parameters:
        model: 模型实例

    Returns:
        bool: 是否为 FAISS 量化器
    """
    return isinstance(model, dict) and "rq" in model


def train_faiss_model(args, data, model_dict, data_loader, accelerator, logger):
    """
    RQKMeans 专用训练函数

    - 不使用 Trainer
    - 直接调用 train_faiss_rq() 训练 FAISS 量化器

    Parameters:
        args: 配置参数
        data: 数据集实例
        model_dict: create_rqkmeans_model() 返回的字典
        data_loader: DataLoader 实例
        accelerator: Accelerator 实例
        logger: 日志记录器

    Returns:
        tuple: (best_loss, best_collision_rate, best_epoch)
    """
    rq = model_dict["rq"]
    codebook_size = model_dict["codebook_size"]
    num_levels = model_dict["num_levels"]

    logger.info("[Train] Starting FAISS ResidualQuantizer training")
    logger.info("[Train] codebook_size=%d, num_levels=%d", codebook_size, num_levels)

    # 收集所有数据为连续 float32 数组
    all_embeddings = []
    for batch in data_loader:
        if isinstance(batch, (tuple, list)):
            _, emb = batch
        else:
            emb = batch
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        all_embeddings.append(emb)

    data_np = np.ascontiguousarray(np.concatenate(all_embeddings, axis=0).astype(np.float32))
    N, d = data_np.shape
    logger.info("[Train] Collected %d samples, dim=%d", N, d)

    # 训练量化器
    rq = train_faiss_rq(
        data=data_np,
        codebook_size=codebook_size,
        num_levels=num_levels,
        verbose=True,
        rq=rq,
    )

    # 可选：Sinkhorn 均匀映射
    if getattr(args, 'uniform_mapping', False):
        logger.info("[Train] Applying Sinkhorn uniform mapping")
        codes_raw = encode_with_rq(rq, data_np, codebook_size)
        codes_bal = sinkhorn_uniform_mapping(
            rq, data_np, codes_raw,
            iters=getattr(args, 'uniform_iters', 30),
            tau=getattr(args, 'uniform_tau', None),
            verbose=True,
        )
        codes_final = codes_bal
    else:
        codes_final = encode_with_rq(rq, data_np, codebook_size)

    # 计算 collision rate
    unique_codes = len(set(map(tuple, codes_final)))
    collision_rate = 1 - unique_codes / N
    best_loss = 0.0
    best_collision_rate = collision_rate
    best_epoch = 1

    logger.info("[Train] FAISS training completed")
    logger.info("[Train] unique_codes=%d, total=%d, collision_rate=%.6f",
                unique_codes, N, collision_rate)

    # 保存 artifacts
    if accelerator.is_main_process and args.ckpt_dir:
        save_codebook_npz(rq, args.ckpt_dir, codebook_size)
        save_faiss_index(rq, args.ckpt_dir)

    # 保存 SID 文件
    if accelerator.is_main_process and getattr(args, 'dump_sids', True):
        logger.info("[Dump] Saving SID index...")
        _dump_sids_faiss(args, codes_final, logger)
        logger.info("[Dump] SID index saved successfully")

    return best_loss, best_collision_rate, best_epoch


def _dump_sids_faiss(args, codes, logger=None):
    """保存 FAISS/RQKMeans 模式的 SID 文件"""
    import json

    N = codes.shape[0]
    num_levels = codes.shape[1]

    if args.eval_dump_root:
        output_dir = args.eval_dump_root
    else:
        output_dir = args.ckpt_dir

    os.makedirs(output_dir, exist_ok=True)

    if getattr(args, 'dump_sids_format', 'json') == 'json':
        codes_plus_one = codes + 1
        sid_json = {}
        for idx in range(N):
            codes_list = [f"<{chr(97 + l)}_{codes_plus_one[idx, l]}>" for l in range(num_levels)]
            sid_json[str(idx)] = codes_list

        json_path = os.path.join(output_dir, "sid_index.json")
        with open(json_path, "w") as f:
            json.dump(sid_json, f, indent=2)
        log_msg = f"[Dump] SID index saved to: {json_path} ({N} items, {num_levels} levels)"
        print(log_msg)
        if logger:
            logger.info(log_msg)
    else:
        npy_path = os.path.join(output_dir, "sid_codes.npy")
        np.save(npy_path, codes)
        log_msg = f"[Dump] SID codes saved to: {npy_path} ({N} items, {num_levels} levels)"
        print(log_msg)
        if logger:
            logger.info(log_msg)


def train_constrained_model(args, data, model_dict, data_loader, accelerator, logger):
    """
    RQKMeans Constrained 专用训练函数

    - 使用 k-means-constrained 库进行平衡聚类
    - 直接调用 residual_kmeans_constrained() 训练

    Parameters:
        args: 配置参数
        data: 数据集实例
        model_dict: create_rqkmeans_constrained_model() 返回的字典
        data_loader: DataLoader 实例
        accelerator: Accelerator 实例
        logger: 日志记录器

    Returns:
        tuple: (best_loss, best_collision_rate, best_epoch)
    """
    codebook_size = model_dict["codebook_size"]
    num_levels = model_dict["num_levels"]

    logger.info(
        "[Train] Starting RQKMeans Constrained training"
    )
    logger.info(
        "[Train] codebook_size=%d, num_levels=%d",
        codebook_size, num_levels
    )

    # 收集所有数据为连续 float32 数组
    all_embeddings = []
    for batch in data_loader:
        if isinstance(batch, (tuple, list)):
            _, emb = batch
        else:
            emb = batch
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        all_embeddings.append(emb)

    data_np = np.ascontiguousarray(
        np.concatenate(all_embeddings, axis=0).astype(np.float32)
    )
    N, d = data_np.shape
    logger.info("[Train] Collected %d samples, dim=%d", N, d)

    # 执行残差K-Means（平衡聚类）
    K_values = [codebook_size] * num_levels
    codes_all, codebooks, recon = residual_kmeans_constrained(
        data_np,
        K=K_values,
        L=num_levels,
        max_iter=args.kmeans_iters,
        tol=getattr(args, 'kmeans_tol', 1e-4),
        random_state=args.seed,
        verbose=True,
    )

    # 计算 collision rate
    codes_T = codes_all.T  # (N, L)
    unique_codes = len(set(map(tuple, codes_T)))
    collision_rate = 1 - unique_codes / N
    best_loss = float(np.mean((data_np - recon) ** 2))
    best_collision_rate = collision_rate
    best_epoch = 1

    logger.info("[Train] Constrained training completed")
    logger.info(
        "[Train] unique_codes=%d, total=%d, collision_rate=%.6f, mse=%.6f",
        unique_codes, N, collision_rate, best_loss
    )

    # 更新 model_dict 中的 codebooks
    model_dict["codebooks"] = codebooks

    # 保存 artifacts
    if accelerator.is_main_process and args.ckpt_dir:
        import json
        os.makedirs(args.ckpt_dir, exist_ok=True)

        # 保存 codebook 到 npz 文件
        codebook_path = os.path.join(
            args.ckpt_dir,
            "codebooks_constrained.npz"
        )
        np.savez_compressed(
            codebook_path,
            **{f"codebook_{i}": cb for i, cb in enumerate(codebooks)}
        )
        logger.info("[Train] Saved codebooks to: %s", codebook_path)

        # 保存 codes 到 npy 文件
        codes_path = os.path.join(args.ckpt_dir, "codes_constrained.npy")
        np.save(codes_path, codes_T)
        logger.info("[Train] Saved codes to: %s", codes_path)

        # 生成并保存 JSON 索引
        codes_plus_one = codes_T + 1  # +1 offset for token format
        codes_json = {}
        for idx, row in enumerate(codes_plus_one):
            codes_list = []
            for level, code in enumerate(row):
                codes_list.append(f"<{chr(97 + level)}_{code}>")
            codes_json[str(idx)] = codes_list

        json_path = os.path.join(args.ckpt_dir, "sid_index.json")
        with open(json_path, "w") as f:
            json.dump(codes_json, f, indent=2)
        logger.info("[Train] Saved SID index to: %s", json_path)

    # 保存 SID 文件（使用统一逻辑）
    if accelerator.is_main_process and getattr(args, 'dump_sids', True):
        logger.info("[Dump] Saving SID index...")
        _dump_sids_faiss(args, codes_T, logger)
        logger.info("[Dump] SID index saved successfully")

    return best_loss, best_collision_rate, best_epoch


# ----------------------------
# 主函数
# ----------------------------
def main():
    # 解析命令行参数
    args = parse_args()

    # 加载 YAML 配置（支持默认配置 + 业务配置的合并）
    config = load_config(args.config, default_config_path="configs/default.yaml")

    # 合并配置和参数（CLI 参数优先）
    args = _merge_config_with_args(config, args)

    # 处理 nargs 参数的类型转换
    if isinstance(args.sk_epsilons, list):
        args.sk_epsilons = [float(x) for x in args.sk_epsilons]
    if isinstance(args.num_emb_list, list):
        args.num_emb_list = [int(x) for x in args.num_emb_list]
    if isinstance(args.layers, list):
        args.layers = [int(x) for x in args.layers]

    # 设置日志（log_dir 从 ckpt_dir 派生）
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_dir = args.ckpt_dir if args.ckpt_dir else None
    log_file = args.log_file if args.log_file else "train_sid.log"

    logger = get_logger(
        name="SID_Gen",
        log_dir=log_dir,
        log_file=log_file,
        level=level,
    )

    set_seed(args.seed)
    accelerator = Accelerator()

    wandb_run = None
    if accelerator.is_main_process and args.use_wandb:
        log_dir = args.ckpt_dir if args.ckpt_dir else None
        if log_dir:
            run_path = os.path.dirname(args.ckpt_dir.rstrip('/'))
            run_name = os.path.basename(run_path)
        else:
            run_name = "default_run"

        if args.data_path:
            data_parent = os.path.dirname(os.path.dirname(args.data_path.rstrip('/')))
            wandb_dir = data_parent
        else:
            wandb_dir = log_dir

        wandb_init_kwargs = {
            "project": args.wandb_project,
            "name": run_name,
            "dir": wandb_dir,
            "mode": "offline",
        }
        if args.wandb_entity:
            wandb_init_kwargs["entity"] = args.wandb_entity
        if args.wandb_tags:
            wandb_init_kwargs["tags"] = args.wandb_tags

        wandb_run = wandb.init(**wandb_init_kwargs)

        hyperparams = {
            "model_type": args.model_type,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_emb_list": args.num_emb_list,
            "e_dim": args.e_dim,
            "layers": args.layers,
            "beta": args.beta,
            "recon_weight": args.recon_weight,
            "quant_loss_weight": args.quant_loss_weight,
            "dropout_prob": args.dropout_prob,
            "bn": args.bn,
            "loss_type": args.loss_type,
            "kmeans_init": args.kmeans_init,
            "sk_epsilons": args.sk_epsilons,
            "sk_iters": args.sk_iters,
            "seed": args.seed,
        }
        wandb_run.config.update(hyperparams)

    if accelerator.is_main_process:
        _log_namespace(logger, args)

    # 加载数据集
    data = create_emb_dataset(
        data_type=args.data_type,
        data_path=args.data_path,
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        columns=args.columns,
        column_mapper=args.column_mapper,
        max_cache_shards=args.max_cache_shards,
        shuffle_shards=args.shuffle_shards,
        prefetch_shards=args.prefetch_shards,
        seed=args.seed,
        norm=args.norm,
        csv_sep=args.csv_sep,
        expected_emb_dim=args.expected_emb_dim,
    )

    if accelerator.is_main_process:
        logger.info("Loaded embeddings dim = %s", data.dim)

    # 创建模型（通过工厂函数根据 model_type 创建）
    model = create_model_by_type(args, data, accelerator, logger)

    # 创建 DataLoader
    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
    )

    # 根据模型类型选择训练方式
    if is_faiss_model(model):
        # RQKMeans 模式：使用 FAISS 独立训练流程
        logger.info("[Train] Using FAISS training pipeline for rqkmeans model")
        best_loss, best_collision_rate, best_epoch = train_faiss_model(
            args, data, model, data_loader, accelerator, logger
        )
    elif args.model_type == "rqkmeans_constrained":
        # RQKMeans Constrained 模式：使用约束聚类训练流程
        logger.info("[Train] Using Constrained K-Means training pipeline")
        best_loss, best_collision_rate, best_epoch = train_constrained_model(
            args, data, model, data_loader, accelerator, logger
        )
    else:
        # RQVAE / RQKMeans+ 模式：使用 Trainer
        model = model.to(args.device)
        trainer = Trainer(args, model, len(data_loader), accelerator=accelerator, wandb_run=wandb_run)
        best_loss, best_collision_rate, best_epoch = trainer.fit(data_loader)

    if accelerator.is_main_process:
        logger.info("Best Loss %s", best_loss)
        logger.info("Best Collision Rate %s", best_collision_rate)
        logger.info("Best Epoch %s", best_epoch)

    if accelerator.is_main_process and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
