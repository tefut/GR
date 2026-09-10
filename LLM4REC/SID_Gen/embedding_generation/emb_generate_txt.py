# -*- coding: utf-8 -*-
"""
用途：
从 Parquet/CSV 文件中读取数据，拼接文本后生成 embedding。
支持 Accelerate 多进程分片，每个 rank 各自保存 shard，最后由主进程合并。

配置：
- 默认配置: configs/default.yaml
- 业务配置: configs/game.yaml
- CLI 参数可以覆盖配置
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoProcessor

# 使用统一的配置加载器
from utils.config_loader import load_yaml_config
from utils.log_utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings with Accelerate")

    # 只保留 --config 参数
    parser.add_argument(
        "--config",
        type=str,
        default="configs/game.yaml",
        help="业务配置文件路径",
    )

    # 输入输出参数
    parser.add_argument("--input_path", type=str, default=None, help="输入 parquet/csv 路径")
    parser.add_argument("--output_path", type=str, default=None, help="输出 npz 路径")

    # 模型参数
    parser.add_argument("--plm_checkpoint", type=str, default=None, help="模型路径")
    parser.add_argument("--batch_size", type=int, default=None, help="推理 batch size")
    parser.add_argument("--max_sent_len", type=int, default=None, help="最大文本长度")
    parser.add_argument("--pooling", type=str, default=None, choices=["mean", "cls", "last"])
    parser.add_argument("--dtype", type=str, default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--norm_embed", action="store_true")
    parser.add_argument("--word_drop_ratio", type=float, default=None)

    # 输出参数
    parser.add_argument("--save_shards", type=int, default=None)
    parser.add_argument("--shard_dir", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)

    # 文本生成参数
    parser.add_argument("--input_columns", nargs='+', default=None, help="输入列名列表")
    parser.add_argument("--separator", type=str, default=None, help="文本拼接分隔符")
    parser.add_argument("--input_sep", type=str, default=',', help="输入文件的列分隔符")
    parser.add_argument("--input_columns_desc", nargs='+', default=None, help="各列的描述")
    parser.add_argument("--attach_desc", action="store_true", help="是否为每个字段附加描述")

    # 多模态参数
    parser.add_argument("--enable_multimodal", action="store_true")
    parser.add_argument("--multimodal_mode", type=str, default=None, choices=["text_only", "text_image"])
    parser.add_argument("--image_mapping_file", type=str, default=None, help="图片映射文件路径(CSV/TXT/JSON)")
    parser.add_argument("--image_base_dir", type=str, default=None, help="图片基础目录")
    parser.add_argument("--load_image_method", type=str, default=None, choices=["file", "dir"],
                        help="图片加载方式: file从mapping文件读取, dir从base_dir目录读取")
    parser.add_argument("--image_dir_mode", type=str, default=None, choices=["single", "multiple"],
                        help="dir模式下的目录结构: single图片直接在该目录, multiple每层文件夹名为id")
    parser.add_argument("--path_sep", type=str, default=None, help="CSV/TXT文件中img_path列的分隔符")
    parser.add_argument("--image_path_col", type=str, default=None, help="CSV/TXT文件中的img_path列名")

    # 日志参数
    parser.add_argument("--log_level", type=str, default=None)

    return parser.parse_args()


def _merge_config_with_args(config: Dict, args) -> argparse.Namespace:
    """
    将 YAML 配置与 CLI 参数合并，CLI 参数优先
    从新的配置结构加载各模块参数
    """
    # 从配置获取各模块的默认值
    text_gen_cfg = config.get("text_generation", {})
    model_cfg = config.get("model", {})
    output_cfg = config.get("output", {})
    image_mapping_cfg = config.get("image_mapping", {})
    col_mapping = config.get("column_mapping", {})

    # 创建一个新的 Namespace 来存储合并后的参数
    merged = argparse.Namespace()

    # === text_generation 配置 ===
    merged.input_columns = args.input_columns if args.input_columns else text_gen_cfg.get("input_columns", [])
    merged.separator = args.separator if args.separator else text_gen_cfg.get("separator", " ")
    merged.attach_desc = args.attach_desc or text_gen_cfg.get("attach_desc", False)
    merged.input_columns_desc = args.input_columns_desc if args.input_columns_desc else text_gen_cfg.get(
        "input_columns_desc", [])
    merged.empty_placeholder = text_gen_cfg.get("empty_placeholder", "")

    # === model 配置 ===
    merged.plm_checkpoint = args.plm_checkpoint if args.plm_checkpoint else model_cfg.get("plm_checkpoint", "")
    merged.batch_size = args.batch_size if args.batch_size else model_cfg.get("batch_size", 64)
    merged.max_sent_len = args.max_sent_len if args.max_sent_len else model_cfg.get("max_sent_len", 512)
    merged.pooling = args.pooling if args.pooling else model_cfg.get("pooling", "mean")
    merged.dtype = args.dtype if args.dtype else model_cfg.get("dtype", "float16")
    merged.norm_embed = args.norm_embed or model_cfg.get("norm_embed", False)
    merged.word_drop_ratio = args.word_drop_ratio if args.word_drop_ratio is not None else model_cfg.get(
        "word_drop_ratio", -1.0)

    # === output 配置 ===
    merged.input_path = args.input_path if args.input_path else output_cfg.get("input_path", "")
    merged.output_path = args.output_path if args.output_path else output_cfg.get("output_path", "")
    merged.save_shards = args.save_shards if args.save_shards is not None else output_cfg.get("save_shards", 1)
    merged.shard_dir = args.shard_dir if args.shard_dir else output_cfg.get("shard_dir", "")
    merged.save_csv = output_cfg.get("save_csv", False)
    merged.log_file = args.log_file if args.log_file else output_cfg.get("log_file", "")

    # === image_mapping 配置 ===
    merged.image_mapping_enabled = image_mapping_cfg.get("enabled", False)
    merged.image_mapping_file = args.image_mapping_file if args.image_mapping_file else image_mapping_cfg.get(
        "mapping_file", "")
    merged.image_base_dir = args.image_base_dir if args.image_base_dir else image_mapping_cfg.get("base_dir", "")
    merged.load_image_method = args.load_image_method if args.load_image_method else image_mapping_cfg.get(
        "load_image_method", "file")
    merged.image_dir_mode = args.image_dir_mode if args.image_dir_mode else image_mapping_cfg.get(
        "dir_mode", "single")
    merged.path_sep = args.path_sep if args.path_sep else image_mapping_cfg.get("path_sep", ",")
    if args.image_path_col:
        merged.image_path_col = args.image_path_col
    else:
        merged.image_path_col = image_mapping_cfg.get("img_path_col", "img_path")

    # === 多模态配置 ===
    merged.enable_multimodal = args.enable_multimodal or model_cfg.get("enable_multimodal", False)
    merged.multimodal_mode = args.multimodal_mode if args.multimodal_mode else model_cfg.get("multimodal_mode",
                                                                                             "text_only")
    merged.input_sep = args.input_sep

    # === 日志配置 ===
    merged.log_level = args.log_level if args.log_level else config.get("log_level", "INFO")

    # === column_mapping 配置 ===
    merged.column_mapping = col_mapping

    # === 主键列名（从 column_mapping 获取） ===
    merged.primary_key = col_mapping.get("primary_key", "id")

    # 保存原始配置引用
    merged._config = config
    merged.config_path = args.config

    return merged


def clean_text(text) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and np.isnan(text):
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()


def get_torch_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _get_primary_key_from_mapping(args) -> str:
    """获取主键列名"""
    column_mapping = getattr(args, 'column_mapping', None)
    if column_mapping:
        return column_mapping.get("primary_key", "id")
    return "id"


def _get_column_from_mapping(args, key: str, default: str) -> str:
    """从配置中获取列名映射"""
    column_mapping = getattr(args, 'column_mapping', None)
    if column_mapping:
        columns = column_mapping.get("columns", {})
        return columns.get(key, default)
    return default


def validate_args(args):
    """验证配置文件存在"""
    if not args.config_path:
        raise ValueError("配置文件路径不能为空")
    if not os.path.exists(args.config_path):
        raise ValueError(f"配置文件不存在: {args.config_path}")
    return args


def read_input_data(input_path: str, logger: logging.Logger, sep: str) -> pd.DataFrame:
    """
    读取输入数据文件，支持 parquet 和 csv 格式
    返回 DataFrame
    """
    logger.info("Reading input data: %s", input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 根据后缀判断文件类型
    if input_path.lower().endswith('.csv'):
        df = pd.read_csv(input_path)
        logger.info("Loaded CSV file, shape=%s", str(df.shape))
    elif input_path.lower().endswith('.txt'):
        df = pd.read_csv(input_path, encoding='utf-8-sig', sep='\x01', quoting=csv.QUOTE_NONE, engine='python')
        logger.info("Loaded txt file, shape=%s", str(df.shape))
    else:
        df = pd.read_parquet(input_path)
        logger.info("Loaded Parquet file, shape=%s", str(df.shape))

    logger.info("Columns=%s", list(df.columns))

    return df


def build_app_text(
        row: pd.Series,
        input_columns: List[str],
        separator: str = " ",
        input_columns_desc: List[str] = None,
        attach_desc: bool = False,
        empty_placeholder: str = ""
) -> str:
    """
    根据配置拼接文本
    支持通用配置：input_columns, separator, input_columns_desc, attach_desc, empty_placeholder
    空值会被跳过，不添加到结果中
    """
    parts: List[str] = []

    for i, col in enumerate(input_columns):
        v = clean_text(row.get(col, ""))

        # 跳过空值
        if not v:
            continue

        # 附加描述
        if attach_desc and input_columns_desc and i < len(input_columns_desc):
            desc = input_columns_desc[i]
            if desc:
                v = f"{desc}: {v}"

        parts.append(v)

    if not parts:
        return "unknown item"

    return separator.join(parts)


def load_image_mapping(
        mapping_file: str,
        base_dir: str = "",
        path_sep: str = ",",
        id_col: str = "id",
        img_path_col: str = "img_path"
) -> Dict[str, List[str]]:
    """
    加载图片映射文件，支持 JSON/CSV/TXT 格式
    JSON格式：{"id1": ["path1", "path2"], "id2": ["path3"], ...}
    CSV/TXT格式：需要有id列和img_path列，img_path列中多个路径用path_sep分隔
    支持 base_dir 拼接
    """
    if not mapping_file or not os.path.exists(mapping_file):
        return {}

    if mapping_file.lower().endswith('.json'):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
    elif mapping_file.lower().endswith(('.csv', '.txt')):
        df = pd.read_csv(mapping_file, encoding='utf-8-sig', sep='\x01', quoting=csv.QUOTE_NONE,
                         engine='python') if mapping_file.lower().endswith('.txt') else pd.read_csv(mapping_file)
        mapping = {}
        for _, row in df.iterrows():
            item_id = str(row.get(id_col, ""))
            if not item_id:
                continue
            img_paths_str = str(row.get(img_path_col, ""))
            if img_paths_str and img_paths_str != "nan":
                paths = img_paths_str.split(path_sep)
                mapping[item_id] = [p.strip() for p in paths if p.strip()]
            else:
                mapping[item_id] = []
    else:
        return {}

    if base_dir:
        result = {}
        for item_id, paths in mapping.items():
            if isinstance(paths, list):
                result[item_id] = [os.path.join(base_dir, p) if not os.path.isabs(p) else p for p in paths]
            elif isinstance(paths, str):
                path = paths
                if not os.path.isabs(path):
                    path = os.path.join(base_dir, path)
                result[item_id] = [path]
            else:
                result[item_id] = []
        return result

    return mapping


def load_image_from_dir(base_dir: str, mode: str = "single") -> Dict[str, List[str]]:
    """
    从base_dir直接读取图片
    mode:
        - single: 所有图片直接在base_dir，文件名为id（不含后缀），每个id对应一个图片
        - multiple: base_dir下有多个文件夹，文件夹名为id，文件夹下有图片
    返回格式与load_image_mapping一致：Dict[str, List[str]]
    """
    if not base_dir or not os.path.exists(base_dir):
        return {}

    result = {}
    supported_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}

    if mode == "single":
        for fname in os.listdir(base_dir):
            fpath = os.path.join(base_dir, fname)
            if not os.path.isfile(fpath):
                continue
            _, ext = os.path.splitext(fname)
            if ext.lower() not in supported_exts:
                continue
            item_id = os.path.splitext(fname)[0]
            result[item_id] = [fpath]

    elif mode == "multiple":
        for dir_name in os.listdir(base_dir):
            dir_path = os.path.join(base_dir, dir_name)
            if not os.path.isdir(dir_path):
                continue
            images = []
            for fname in os.listdir(dir_path):
                fpath = os.path.join(dir_path, fname)
                if os.path.isfile(fpath):
                    _, ext = os.path.splitext(fname)
                    if ext.lower() in supported_exts:
                        images.append(fpath)
            if images:
                result[dir_name] = images

    return result


def preprocess_data(args, logger: logging.Logger):
    """
    预处理数据，返回格式根据是否开启多模态而定：
    - 非多模态模式：List[Tuple[str, str]] = [(item_id, text), ...]
    - 多模态模式：List[Tuple[str, Dict]] = [(item_id, {"text": ..., "images": [...]}), ...]
    """
    df = read_input_data(args.input_path, logger, sep=args.input_sep)

    # 从配置获取主键列名
    pk_col = _get_primary_key_from_mapping(args)

    # 加载图片映射
    image_mapping = {}
    enable_multimodal = getattr(args, 'enable_multimodal', False)
    multimodal_mode = getattr(args, 'multimodal_mode', 'text_only')

    if enable_multimodal and multimodal_mode == "text_image":
        load_image_method = getattr(args, 'load_image_method', 'file')
        if load_image_method == "dir":
            image_dir_mode = getattr(args, 'image_dir_mode', 'single')
            image_mapping = load_image_from_dir(args.image_base_dir, mode=image_dir_mode)
            logger.info("Loaded image mapping from dir (mode=%s): %d items", image_dir_mode, len(image_mapping))
        else:
            path_sep = getattr(args, 'path_sep', ',')
            image_path_col = getattr(args, 'image_path_col', 'img_path')
            image_mapping = load_image_mapping(
                args.image_mapping_file,
                args.image_base_dir,
                path_sep=path_sep,
                img_path_col=image_path_col,
                id_col=pk_col
            )
            logger.info("Loaded image mapping: %d items", len(image_mapping))

    missing_image_count = 0
    out = []

    for _, row in df.iterrows():
        item_id = clean_text(row.get(pk_col, ""))
        if not item_id:
            continue

        text = build_app_text(
            row=row,
            input_columns=args.input_columns,
            separator=args.separator,
            input_columns_desc=args.input_columns_desc,
            attach_desc=args.attach_desc,
            empty_placeholder=args.empty_placeholder
        )

        if enable_multimodal and multimodal_mode == "text_image" and image_mapping:
            # 文本+图片混合模式
            images = image_mapping.get(item_id, [])
            # 过滤存在的图片
            existing_images = [img for img in images if os.path.exists(img)]

            if existing_images:
                out.append((item_id, {"text": text, "images": existing_images}))
            else:
                # 图片缺失，只传文本
                missing_image_count += 1
                out.append((item_id, {"text": text}))
        else:
            # 纯文本模式
            out.append((item_id, text))

    if not out:
        logger.warning("No valid rows found in input data.")
    else:
        logger.info("Total valid items=%d", len(out))

    if enable_multimodal and multimodal_mode == "text_image" and image_mapping:
        logger.info("Missing images count: %d / %d", missing_image_count, len(out))

    return out


def load_model(model_path: str, dtype: str, logger: logging.Logger, enable_multimodal: bool = False):
    logger.info("Loading model from: %s", model_path)

    torch_dtype = get_torch_dtype(dtype)

    # 使用普通的文本embedding模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    return tokenizer, model


def pooling_hidden(outputs, attention_mask, pooling: str):
    last_hidden = outputs.last_hidden_state

    if pooling == "cls":
        return last_hidden[:, 0, :]
    elif pooling == "mean":
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    elif pooling == "last":
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            return last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    else:
        raise NotImplementedError


def generate_app_embedding(
        args,
        app_data_list,
        tokenizer,
        model,
        accelerator: Accelerator,
        logger: logging.Logger,
):
    """
    生成embedding。app_data_list格式：
    - 非多模态模式：List[Tuple[str, str]] = [(item_id, text), ...]
    - 多模态模式：List[Tuple[str, Dict]] = [(item_id, {"text": ..., "images": [...]}), ...]
    """
    if not app_data_list:
        if accelerator.is_main_process:
            logger.warning("Empty app_data_list; nothing to do.")
        return

    # 判断是否为多模态模式（根据第二个元素的类型判断）
    is_multimodal = isinstance(app_data_list[0][1], dict)

    if is_multimodal and args.enable_multimodal:
        # 多模态模式：使用VL模型
        processor = AutoProcessor.from_pretrained(args.plm_checkpoint)
        generate_app_embedding_multimodal(
            args=args,
            app_data_list=app_data_list,
            model=model,
            processor=processor,
            accelerator=accelerator,
            logger=logger,
        )
    else:
        # 纯文本模式：使用普通文本embedding模型
        generate_app_embedding_text(
            args=args,
            app_data_list=app_data_list,
            tokenizer=tokenizer,
            model=model,
            accelerator=accelerator,
            logger=logger,
        )


def generate_app_embedding_text(
        args,
        app_data_list,
        tokenizer,
        model,
        accelerator: Accelerator,
        logger: logging.Logger,
):
    """纯文本embedding生成（使用普通文本模型）"""
    # 使用 accelerator.split_between_processes 自动切分数据
    with accelerator.split_between_processes(app_data_list) as local_data:
        local_data = list(local_data)
    first_data = local_data[0]
    logger.info("first_sample %s=%s", _get_primary_key_from_mapping(args), first_data[0])
    logger.info("first_sample text=%s", first_data[1][:1000])

    total_items = len(app_data_list)
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    if accelerator.is_main_process:
        logger.info("Total items=%d", total_items)
        logger.info("Start generating text embeddings with %d processes", num_processes)

    if not local_data:
        logger.info("[rank%d] No data to process", process_index)
        local_results = []
    else:
        local_ids, local_texts = zip(*local_data)
        logger.info(
            "[rank%d] local size=%d",
            process_index,
            len(local_texts),
        )

        local_results: List[Tuple[str, np.ndarray]] = []
        batch_size = args.batch_size
        max_sent_len = args.max_sent_len
        word_drop_ratio = args.word_drop_ratio

        pbar = tqdm(
            total=len(local_texts),
            desc="Proc %d" % process_index,
            disable=not accelerator.is_local_main_process,
        )

        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        with torch.no_grad():
            for i in range(0, len(local_texts), batch_size):
                batch_texts = list(local_texts[i:i + batch_size])
                batch_ids = local_ids[i:i + batch_size]

                if word_drop_ratio > 0:
                    processed = []
                    for text in batch_texts:
                        sent = text.split(" ")
                        new_sent = [wd for wd in sent if random.random() > word_drop_ratio]
                        new_text = " ".join(new_sent).strip()
                        if not new_text:
                            new_text = "unknown item"
                        processed.append(new_text)
                    batch_texts = processed

                encoded = tokenizer(
                    batch_texts,
                    max_length=max_sent_len,
                    truncation=True,
                    return_tensors="pt",
                    padding=True,
                ).to(accelerator.device)

                outputs = model(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                )

                pooled = pooling_hidden(outputs, encoded.attention_mask, args.pooling)
                if args.norm_embed:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                batch_emb = pooled.detach().cpu().numpy()

                for idx, emb in zip(batch_ids, batch_emb):
                    local_results.append((str(idx), emb))

                pbar.update(len(batch_texts))

        pbar.close()
        accelerator.wait_for_everyone()

    _save_and_merge_shards(
        args=args,
        local_results=local_results,
        model=model,
        accelerator=accelerator,
        logger=logger,
    )


def generate_app_embedding_multimodal(
        args,
        app_data_list,
        model,
        processor,
        accelerator: Accelerator,
        logger: logging.Logger,
):
    """多模态embedding生成（使用VL模型）"""

    # 数据切分
    with accelerator.split_between_processes(app_data_list) as local_data:
        local_data = list(local_data)
    first_data = local_data[0]
    logger.info("first_sample %s=%s", _get_primary_key_from_mapping(args), first_data[0])
    logger.info("first_sample text=%s", first_data[1]['text'][:1000])
    total_items = len(app_data_list)
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    if accelerator.is_main_process:
        logger.info("Total items=%d", total_items)
        logger.info("Start generating multimodal embeddings with %d processes", num_processes)

    if not local_data:
        logger.info("[rank%d] No data to process", process_index)
        local_results = []
        missing_image_count = 0
    else:
        local_ids, local_data = zip(*local_data)
        logger.info("[rank%d] local size=%d", process_index, len(local_data))

        local_results: List[Tuple[str, np.ndarray]] = []
        batch_size = args.batch_size
        word_drop_ratio = args.word_drop_ratio
        missing_image_count = 0

        pbar = tqdm(
            total=len(local_data),
            desc="Proc %d" % process_index,
            disable=not accelerator.is_local_main_process,
        )

        model = model.to(accelerator.device)
        model.eval()

        with torch.no_grad():
            for i in range(0, len(local_data), batch_size):
                batch_data = local_data[i:i + batch_size]
                batch_ids = local_ids[i:i + batch_size]

                batch_messages = []
                batch_images = []

                for item in batch_data:
                    text = item.get("text", "")
                    images = item.get("images", [])

                    # 丢词处理
                    if word_drop_ratio > 0:
                        sent = text.split(" ")
                        new_sent = [wd for wd in sent if random.random() > word_drop_ratio]
                        text = " ".join(new_sent).strip() or "unknown item"

                    # 支持多图片（当前版本取第一张，接口已支持多图扩展）
                    if images:
                        # 过滤存在的图片
                        valid_images = [img for img in images if os.path.exists(img)]
                        if valid_images:
                            # 构造消息（取第一张图片）
                            content = []
                            for img_path in valid_images:
                                content.append({"type": "image", "image": img_path})
                            content.append({"type": "text", "text": text})
                            msg = {"role": "user", "content": content}
                            batch_messages.append(msg)
                            batch_images.append(valid_images)
                        else:
                            msg = {"role": "user", "content": [{"type": "text", "text": text}]}
                            batch_messages.append(msg)
                            missing_image_count += 1
                    else:
                        msg = {"role": "user", "content": [{"type": "text", "text": text}]}
                        batch_messages.append(msg)
                        missing_image_count += 1

                # 包装成 List[List[dict]]
                if batch_images:
                    # Step 1: 转为字符串列表
                    text_strs = processor.apply_chat_template(
                        [[msg] for msg in batch_messages],  # 核心修复
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    # Step 2: processor 统一编码
                    processed = processor(
                        text=text_strs,
                        images=batch_images,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=args.max_sent_len,
                    )
                else:
                    # 纯文本分支
                    text_strs = [msg["content"][0]["text"] for msg in batch_messages]
                    processed = processor.tokenizer(
                        text_strs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=args.max_sent_len,
                    )
                    if "attention_mask" not in processed:
                        processed["attention_mask"] = torch.ones_like(processed["input_ids"])

                # 移动到设备
                processed = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                             for k, v in processed.items()}

                # 调试输出
                if accelerator.is_local_main_process and i == 0:
                    text_strs_len = len(text_strs) if isinstance(text_strs, list) else 'N/A'
                    print(f"text_strs 类型: {type(text_strs)}, 长度: {text_strs_len}")
                    print(f"第一个text_str {text_strs[0]}")
                    print(f"input_ids shape: {processed['input_ids'].shape}")
                    if 'pixel_values' in processed:
                        print(f"pixel_values shape: {processed['pixel_values'].shape}")
                    # 安全获取 image_token_id
                    image_token = getattr(processor, "image_token", "<|image|>")
                    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
                    img_mask = processed["input_ids"] == image_token_id
                    print(f"每个样本的 image token 数: {img_mask.sum(dim=1).tolist()}")

                # 前向传播 + embedding 提取
                outputs = model(**processed)
                attention_mask = processed.get('attention_mask')
                pooled = pooling_hidden(outputs, attention_mask, args.pooling)
                if args.norm_embed:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                batch_emb = pooled.detach().cpu().numpy()

                for idx, emb in zip(batch_ids, batch_emb):
                    local_results.append((str(idx), emb))

                pbar.update(len(batch_data))

        pbar.close()
        accelerator.wait_for_everyone()

    logger.info("[rank%d] missing images count: %d", process_index, missing_image_count)

    _save_and_merge_shards(
        args=args,
        local_results=local_results,
        model=model,
        accelerator=accelerator,
        logger=logger,
    )


def convert_float2str(embedding):
    """convert a float embedding to a list of str with sep ,"""
    embedding_str = [str(v) for v in embedding]
    return ','.join(embedding_str)


def _save_and_merge_shards(
        args,
        local_results: List[Tuple[str, np.ndarray]],
        model,
        accelerator: Accelerator,
        logger: logging.Logger,
):
    """保存分片并合并结果"""
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    local_results.sort(key=lambda x: x[0])

    # 获取主键列名
    pk_col = _get_primary_key_from_mapping(args)

    # 使用动态字段名
    primary_key_name = pk_col

    if local_results:
        shard_ids = np.array([x[0] for x in local_results], dtype=object)
        shard_emb = np.stack([x[1] for x in local_results], axis=0)
    else:
        shard_ids = np.array([], dtype=object)
        hidden_size = getattr(model.config, "hidden_size", 0)
        shard_emb = np.empty((0, hidden_size), dtype=np.float32)

    if args.save_shards == 1:
        if accelerator.is_main_process:
            os.makedirs(args.shard_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        shard_path = os.path.join(args.shard_dir, f"part_rank{process_index:03d}.npz")
        np.savez(shard_path, **{primary_key_name: shard_ids, "embedding": shard_emb})
        logger.info("[rank%d] saved shard: %s shape=%s", process_index, shard_path, str(shard_emb.shape))

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("Merging shards on main process ...")

        all_pairs: List[Tuple[str, np.ndarray]] = []

        if args.save_shards == 1:
            for r in range(num_processes):
                p = os.path.join(args.shard_dir, f"part_rank{r:03d}.npz")
                if not os.path.exists(p):
                    logger.warning("Shard not found: %s", p)
                    continue

                data = np.load(p, allow_pickle=True)
                ids = data[primary_key_name]
                emb = data["embedding"]

                for cur_id, cur_emb in zip(ids, emb):
                    all_pairs.append((str(cur_id), cur_emb))
        else:
            logger.warning("save_shards=0 时当前版本无法跨 rank 合并，建议设为 1")
            return

        all_pairs.sort(key=lambda x: x[0])

        if not all_pairs:
            logger.warning("No merged results found.")
            return

        final_ids = np.array([x[0] for x in all_pairs], dtype=object)
        final_embeddings = np.stack([x[1] for x in all_pairs], axis=0)

        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        np.savez(args.output_path, **{primary_key_name: final_ids, "embedding": final_embeddings})
        logger.info("Saved merged embeddings to %s shape=%s", args.output_path, str(final_embeddings.shape))

        print("\n========== merged result preview ==========")
        print(f"{primary_key_name} shape:", final_ids.shape)
        print("embedding shape:", final_embeddings.shape)

        preview_n = min(10, len(final_ids))
        for i in range(preview_n):
            print(f"[{i}] {primary_key_name}={final_ids[i]}")
            print(f"    embedding[:10]={final_embeddings[i][:10]}")
        print("===========================================\n")

        if getattr(args, 'save_csv', False):
            csv_data = {
                primary_key_name: final_ids,
                'embedding': [convert_float2str(embedding) for embedding in final_embeddings]
            }
            df_csv = pd.DataFrame(csv_data)
            csv_path = os.sep + os.path.join(*args.output_path.split(os.sep)[:-1], f'{primary_key_name}2embedding.csv')
            df_csv.to_csv(csv_path, encoding='utf-8-sig', sep='|', index=False, header=False)
            logger.info("Saved CSV embeddings to %s", csv_path)


def main():
    args = parse_args()

    # 加载 YAML 配置（默认配置 + 业务配置）
    default_config_path = "configs/default.yaml"
    business_config_path = args.config

    print(f"加载配置文件:")
    print(f"  - 默认配置: {default_config_path}")
    print(f"  - 业务配置: {business_config_path}")

    config = load_yaml_config(business_config_path, default_config_path)

    # 合并配置与 CLI 参数（CLI 参数优先）
    args = _merge_config_with_args(config, args)

    # 获取主键列名
    pk_col = _get_primary_key_from_mapping(args)

    print(f"\n合并后的配置:")
    print(f"  input_path: {args.input_path}")
    print(f"  output_path: {args.output_path}")
    print(f"  plm_checkpoint: {args.plm_checkpoint}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  input_columns: {args.input_columns}")
    print(f"  primary_key: {pk_col}")

    args = validate_args(args)

    # 从 output_path 派生日志目录
    log_dir = os.path.dirname(args.output_path) if args.output_path else None
    log_file = args.log_file if args.log_file else "emb_generate.log"

    log = get_logger(
        name="emb_generate_txt",
        log_dir=log_dir,
        log_file=log_file,
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    accelerator = Accelerator()

    if accelerator.is_main_process:
        log.info("Config loaded from: %s", args._config)
        log.info("Running with %d processes", accelerator.num_processes)
        log.info("Args:")
        for k, v in vars(args).items():
            if not k.startswith('_'):
                log.info("  %s = %s", k, v)

    app_data_list = preprocess_data(args, log)

    # 加载模型，根据是否开启多模态选择不同模型
    tokenizer, model = load_model(args.plm_checkpoint, args.dtype, log, args.enable_multimodal)

    # 将模型移动到设备上（非多模态模式）
    # 多模态模式（VL模型）在内部会自动处理设备
    model = model.to(accelerator.device)
    model.eval()

    generate_app_embedding(
        args=args,
        app_data_list=app_data_list,
        tokenizer=tokenizer,
        model=model,
        accelerator=accelerator,
        logger=log,
    )


if __name__ == "__main__":
    main()
