# -*- coding: utf-8 -*-
"""
RQVAE 推理脚本
使用已训练的语义ID模型，推理生成语义ID，保存为CSV文件并计算冲突率

用法:
    python inference_sid.py --config configs/inference_sid.yaml
    python inference_sid.py --config configs/inference_sid.yaml
    --ckpt_dir /path/to/checkpoint --input_file /path/to/input.npz
"""

import argparse
import logging
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from SID_Gen.models.faiss_rq import (
    encode_with_rq,
    load_pretrained_codebook,
    load_faiss_quantizer,
)
from SID_Gen.models.rqkmeans_constrained import residual_kmeans_constrained
from SID_Gen.models.rqkmeans_plus import (
    ResidualEncoderWrapper,
    apply_rqkmeans_plus_strategy,
)
from SID_Gen.models.rqvae import RQVAE
from SID_Gen.my_datasets.emb_datasets import create_emb_dataset
from SID_Gen.utils.config_loader import load_yaml_config
from SID_Gen.utils.log_utils import get_logger
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from torch.utils.data import DataLoader

FAISS_AVAILABLE = True

# ----------------------------
# 常量定义
# ----------------------------
SUPPORTED_MODEL_TYPES = ["rqvae", "rqkmeans", "rqkmeans_plus", "rqkmeans_constrained"]


def _log_namespace(logger: logging.Logger, args):
    """统一用 %s 打印所有参数"""
    logger.info("=================================================")
    for k in sorted(args.__dict__.keys()):
        logger.info("%s = %s", k, getattr(args, k))
    logger.info("=================================================")


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

    best_fn, best_col = min(files, key=lambda x: x[1])
    best_path = os.path.join(ckpt_dir, best_fn)
    logger.info(
        "[Init] Auto-selected ckpt (no preferred prefix matched): %s (collision=%.6f)",
        best_path, best_col
    )
    return best_path, best_col


def load_model(args, accelerator, logger):
    """
    加载预训练模型
    args.pretrained_ckpt:
      - None/""：不加载
      - 具体文件路径：按原逻辑加载
      - 目录路径：自动选 collision 最小的 best_collision_*（找不到则降级）
    """
    if not getattr(args, "pretrained_ckpt", None):
        raise ValueError("pretrained_ckpt is required for inference")

    ckpt_input = args.pretrained_ckpt

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

        obj_list = [ckpt_path]
        broadcast_object_list(obj_list)
        ckpt_path = obj_list[0]
    else:
        ckpt_path = ckpt_input

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

    return sd


def load_config(config_path: str, default_config_path: str = "configs/default.yaml") -> dict:
    """加载 YAML 配置（支持默认配置 + 业务配置的合并）"""
    return load_yaml_config(config_path, default_config_path)


def _merge_config_with_args(config: dict, args) -> argparse.Namespace:
    """
    将 YAML 配置转换为 Namespace
    """
    infer_cfg = config.get("inference_sid", config.get("train_sid", {}))

    merged = argparse.Namespace()

    merged.data_type = infer_cfg.get("data_type", "npz")
    merged.data_path = infer_cfg.get("data_path", "")
    merged.data_dir = infer_cfg.get("data_dir", "")
    merged.manifest_path = infer_cfg.get("manifest_path", "")
    merged.columns = infer_cfg.get("columns", None)
    merged.pretrained_ckpt = infer_cfg.get("pretrained_ckpt", "")
    merged.pretrained_codebook_path = infer_cfg.get("pretrained_codebook_path", "")
    merged.output_file = infer_cfg.get("output_file", "sid_output.csv")
    merged.log_file = infer_cfg.get("log_file", "inference_sid.log")

    merged.batch_size = infer_cfg.get("batch_size", 1024)
    merged.num_workers = infer_cfg.get("num_workers", 4)

    merged.norm = infer_cfg.get("norm", False)
    merged.csv_sep = infer_cfg.get("csv_sep", ",")
    merged.expected_emb_dim = infer_cfg.get("expected_emb_dim", None)
    merged.output_sep = infer_cfg.get("output_sep", ",")

    merged.num_emb_list = infer_cfg.get("num_emb_list", [256, 256, 256])
    merged.e_dim = infer_cfg.get("e_dim", 32)
    merged.layers = infer_cfg.get("layers", [2048, 1024, 512, 256, 128, 64])
    merged.dropout_prob = infer_cfg.get("dropout_prob", 0.0)
    merged.bn = infer_cfg.get("bn", False)
    merged.loss_type = infer_cfg.get("loss_type", "mse")
    merged.kmeans_init = infer_cfg.get("kmeans_init", False)
    merged.kmeans_iters = infer_cfg.get("kmeans_iters", 100)
    merged.sk_epsilons = infer_cfg.get("sk_epsilons", [0.0, 0.0, 0.0])
    merged.sk_iters = infer_cfg.get("sk_iters", 50)
    merged.quant_loss_weight = infer_cfg.get("quant_loss_weight", 1.0)
    merged.recon_weight = infer_cfg.get("recon_weight", 1.0)
    merged.beta = infer_cfg.get("beta", 0.25)

    merged.enable_dead_code_reset = infer_cfg.get("enable_dead_code_reset", False)
    merged.reset_threshold = infer_cfg.get("reset_threshold", 1.0)
    merged.reset_freq = infer_cfg.get("reset_freq", 100)
    merged.ema_decay = infer_cfg.get("ema_decay", 0.99)

    merged.device = config.get("device", infer_cfg.get("device", "npu"))
    merged.seed = infer_cfg.get("seed", 2024)
    merged.log_level = infer_cfg.get("log_level", "INFO")
    merged.model_type = infer_cfg.get("model_type", "rqvae")

    merged.column_mapper = config.get("column_mapping", {})

    return merged


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RQVAE 推理脚本')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/inference_sid.yaml',
        help='配置文件路径（YAML格式）'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        help=f'模型类型: rqvae / rqkmeans / rqkmeans_plus / rqkmeans_constrained'
    )

    parser.add_argument(
        '--pretrained_codebook_path',
        type=str,
        default=None,
        help='预训练 codebook 路径（rqkmeans/rqkmeans_plus 模式专用）'
    )

    return parser.parse_args()


@torch.no_grad()
def inference(args, model, data_loader, accelerator, logger):
    """
    执行推理，生成语义ID并保存到CSV文件
    支持 rqvae/rqkmeans/rqkmeans_plus/rqkmeans_constrained 四种模式
    """
    is_faiss = is_faiss_model(model)

    is_rqkmeans_constrained = isinstance(model, dict) and "codebooks" in model

    if is_faiss:
        logger.info("[Inference] Using FAISS encode mode")
    elif is_rqkmeans_constrained:
        logger.info("[Inference] Using balance encoding (rqkmeans_constrained)")
    else:
        model.eval()

    if accelerator.is_main_process:
        logger.info("[Inference] Start inference with %d batches", len(data_loader))

    all_item_ids = []
    all_sids = []
    local_indices_set = set()
    local_num_sample = 0

    for batch_idx, batch in enumerate(data_loader):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            batch_ids, data = batch
        else:
            batch_ids, data = None, batch

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if hasattr(data, "to"):
            data = data.to(accelerator.device)

        if is_rqkmeans_constrained:
            data_np = np.ascontiguousarray(data.detach().cpu().numpy().astype(np.float32))
            indices = _encode_with_balance_constrained(data_np, model)
            indices = indices.astype(np.int32)
        elif is_faiss:
            data_np = np.ascontiguousarray(data.detach().cpu().numpy().astype(np.float32))
            indices = encode_with_rq(model["rq"], data_np, model["codebook_size"])
            indices = indices.astype(np.int32)
        else:
            indices = model.get_indices(data)
            indices = indices.view(-1, indices.shape[-1]).detach().cpu().numpy()

        bs = int(data.shape[0])
        local_num_sample += bs

        for row in indices:
            sid_str = "-".join([str(int(x)) for x in row])
            local_indices_set.add(sid_str)

        if batch_ids is not None:
            for i, item_id in enumerate(batch_ids):
                sid = "-".join([str(int(x)) for x in indices[i]])
                all_item_ids.append(item_id)
                all_sids.append(sid)
        else:
            for i in range(bs):
                sid = "-".join([str(int(x)) for x in indices[i]])
                all_item_ids.append(f"item_{local_num_sample - bs + i}")
                all_sids.append(sid)

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(data_loader):
            if accelerator.is_main_process:
                progress = (batch_idx + 1) / len(data_loader) * 100
                logger.info(
                    "[Inference][%d/%d][%.1f%%] processed_samples: %d",
                    batch_idx + 1, len(data_loader), progress, local_num_sample
                )

    local_unique_sid = len(local_indices_set)

    if accelerator.is_main_process:
        logger.info(
            "[Inference][local] local_num_sample=%d local_unique_sid=%d",
            local_num_sample,
            local_unique_sid,
        )

    from accelerate.utils import gather_object

    gathered_sets = gather_object([local_indices_set])
    gathered_nums = gather_object([local_num_sample])
    gathered_uniqs = gather_object([local_unique_sid])

    if accelerator.is_main_process:
        global_set = set()
        for s in gathered_sets:
            global_set |= set(s)

        global_num = int(sum(gathered_nums))
        global_unique_sid = int(len(global_set))
        collision_rate = (global_num - global_unique_sid) / max(global_num, 1)

        logger.info(
            "[Inference][global] global_num=%d global_unique_sid=%d collision_num=%d collision_rate=%.6f",
            global_num,
            global_unique_sid,
            global_num - global_unique_sid,
            collision_rate,
        )

        gathered_ids = gather_object([all_item_ids])
        gathered_sids = gather_object([all_sids])

        flat_ids = []
        flat_sids = []
        for ids_list, sids_list in zip(gathered_ids, gathered_sids):
            flat_ids.extend(ids_list)
            flat_sids.extend(sids_list)

        df = pd.DataFrame({
            "id": flat_ids,
            "sid": flat_sids
        })

        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(args.output_file, sep=args.output_sep, index=False)
        logger.info("[Inference] Saved SID results to: %s", args.output_file)
        logger.info("[Inference] Total rows: %d", len(df))

        return collision_rate

    return None


def _encode_with_balance_constrained(X, model):
    """
    使用 balance encoding 对输入数据进行编码

    Args:
        X: 输入数据 (N, d)，numpy array
        model: 模型字典，包含 codebooks 和 K

    Returns:
        indices: (N, num_levels) 编码索引
    """
    codebooks = model["codebooks"]
    num_levels = model["num_levels"]
    K = model["K"]

    N = X.shape[0]
    indices = np.zeros((N, num_levels), dtype=np.int32)

    R = X.copy()
    for l in range(num_levels):
        C_l = codebooks[l]
        distances = np.linalg.norm(R[:, np.newaxis, :] - C_l[np.newaxis, :, :], axis=2)
        codes_l = np.argmin(distances, axis=1)
        indices[:, l] = codes_l
        R = R - C_l[codes_l]

    return indices


def _get_value(cli_value, cfg_value, default):
    """
    获取配置值，CLI 参数优先
    
    Args:
        cli_value: 命令行参数值
        cfg_value: YAML 配置值
        default: 默认值
        
    Returns:
        最终使用的值
    """
    if cli_value is not None:
        return cli_value
    return cfg_value if cfg_value is not None else default


def create_rqvae_model(args, data, accelerator, logger):
    """创建 RQVAE 模型"""
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

    sd = load_model(args, accelerator, logger)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if accelerator.is_main_process:
        total = len(model.state_dict())
        loaded = total - len(missing)
        logger.info("[Init] Loaded params: %s/%s (%.2f%%)", loaded, total, loaded / max(total, 1) * 100.0)

        if missing and len(missing) <= 80:
            logger.info("[Init] Missing keys: %s", missing)
        if unexpected and len(unexpected) <= 80:
            logger.info("[Init] Unexpected keys: %s", unexpected)

    return model


def create_rqkmeans_model(args, data, accelerator, logger):
    """创建 FAISS 量化器模型"""
    import faiss

    codebook_size = args.num_emb_list[0]
    num_levels = len(args.num_emb_list)
    nbits = int(np.log2(codebook_size))

    if args.pretrained_codebook_path:
        logger.info("[Init] Loading pretrained FAISS quantizer from: %s", args.pretrained_codebook_path)
        rq = load_faiss_quantizer(args.pretrained_codebook_path)
    else:
        logger.info("[Init] Creating new FAISS ResidualQuantizer (codebook_size=%d, num_levels=%d)",
                    codebook_size, num_levels)
        rq = faiss.ResidualQuantizer(data.dim, num_levels, nbits)

    return {
        "rq": rq,
        "codebook_size": codebook_size,
        "num_levels": num_levels,
        "dim": data.dim,
    }


def create_rqkmeans_plus_model(args, data, accelerator, logger):
    """创建 RQKMeans+ 模型"""
    if not args.pretrained_codebook_path:
        raise ValueError("pretrained_codebook_path is required for rqkmeans_plus model type.")

    model = create_rqvae_model(args, data, accelerator, logger)
    logger.info("[Init] Applying RQKMeans+ strategy")
    model = apply_rqkmeans_plus_strategy(model, args.pretrained_codebook_path, args.device)
    return model


def create_rqkmeans_constrained_model(args, data, accelerator, logger):
    """创建 rqkmeans_constrained 模型"""
    if not args.pretrained_codebook_path:
        raise ValueError("pretrained_codebook_path is required for rqkmeans_constrained model type.")

    codebook_size = args.num_emb_list[0]
    num_levels = len(args.num_emb_list)

    logger.info("[Init] Loading pretrained codebooks from: %s", args.pretrained_codebook_path)
    ckpt = load_ckpt_cpu(args.pretrained_codebook_path)

    if isinstance(ckpt, dict):
        if "codebooks" in ckpt:
            codebooks = ckpt["codebooks"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            codebooks = ckpt["model"].get("codebooks", [])
        else:
            codebooks = ckpt.get("codebooks", [])
    else:
        codebooks = []

    return {
        "codebooks": codebooks,
        "K": codebook_size,
        "num_levels": num_levels,
        "dim": data.dim,
    }


def create_model_by_type(args, data, accelerator, logger):
    """根据 model_type 创建对应的模型实例"""
    model_type = args.model_type

    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Supported types: {SUPPORTED_MODEL_TYPES}")

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
    """判断模型是否为 FAISS 量化器对象"""
    if not isinstance(model, dict):
        return False
    return "rq" in model or "codebooks" in model


def main():
    args = parse_args()

    config = load_config(args.config, default_config_path="configs/default.yaml")
    base_args = _merge_config_with_args(config, args)

    # CLI 参数覆盖（优先级最高）
    base_args.model_type = _get_value(
        args.model_type, base_args.model_type, "rqvae"
    )
    base_args.pretrained_codebook_path = _get_value(
        args.pretrained_codebook_path, base_args.pretrained_codebook_path, ""
    )

    # 使用更新后的 args
    args = base_args

    if isinstance(args.sk_epsilons, list):
        args.sk_epsilons = [float(x) for x in args.sk_epsilons]
    if isinstance(args.num_emb_list, list):
        args.num_emb_list = [int(x) for x in args.num_emb_list]
    if isinstance(args.layers, list):
        args.layers = [int(x) for x in args.layers]

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_dir = os.path.dirname(args.output_file) if args.output_file else None

    logger = get_logger(
        name="SID_Gen",
        log_dir=log_dir,
        log_file=args.log_file,
        level=level,
    )

    from accelerate.utils import set_seed as acc_set_seed
    acc_set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    accelerator = Accelerator()

    if accelerator.is_main_process:
        _log_namespace(logger, args)

    data = create_emb_dataset(
        data_type=args.data_type,
        data_path=args.data_path,
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        columns=args.columns,
        column_mapper=args.column_mapper,
        max_cache_shards=4,
        shuffle_shards=False,
        prefetch_shards=0,
        seed=args.seed,
        norm=args.norm,
        csv_sep=args.csv_sep,
        expected_emb_dim=args.expected_emb_dim,
    )

    if accelerator.is_main_process:
        logger.info("Loaded embeddings dim = %s", data.dim)

    # 使用模型类型路由工厂函数创建模型
    model = create_model_by_type(args, data, accelerator, logger)

    if is_faiss_model(model):
        logger.info("[Init] FAISS model does not need device placement")
    else:
        model = model.to(args.device)
        model = accelerator.prepare(model)

    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
    )

    collision_rate = inference(args, model, data_loader, accelerator, logger)

    if accelerator.is_main_process:
        logger.info("Inference completed. Final collision rate: %.6f", collision_rate)


if __name__ == "__main__":
    main()
