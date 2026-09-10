# -*- coding: utf-8 -*-
"""
merge_app_sid_to_parquet.py

功能：
1. 读取 item2sid.json 映射文件
2. 读取 input_item_info（可选），获取 itemid 到 sid 的映射
3. 如果有 input_item_info，读取 show_col 定义的列，合并 sid
4. 如果无 input_item_info，只输出 itemid -> sid 的映射文件

支持配置化列名映射，实现跨业务复用
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from SID_Gen.eval_sid.sid_eval_unified import parse_sid_to_codes


def parse_sid_to_codes_and_join(s: str, depth: int):
    res = parse_sid_to_codes(s, depth)
    return ','.join(map(str, res))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_item2sid(path: str) -> Dict[str, Optional[str]]:
    """
    读取 item2sid.json
    输入格式：
    {
      "app_id1": ["<a_227>", "<b_177>", "<c_204>"],
      "app_id2": ["<a_135>", "<b_53>", "<c_141>"]
    }

    输出：
    {
      "app_id1": "<a_227><b_177><c_204>",
      "app_id2": "<a_135><b_53><c_141>"
    }
    """
    logging.info("Loading item2sid: %s", path)

    text = Path(path).read_text(encoding="utf-8")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        start = max(0, e.pos - 200)
        end = min(len(text), e.pos + 200)
        snippet = text[start:end]

        logging.error("JSON decode failed!")
        logging.error("line=%s col=%s pos=%s", e.lineno, e.colno, e.pos)
        logging.error("error msg: %s", e.msg)

        print("\n================ BAD JSON CONTEXT ================\n")
        print(snippet)
        print("\n==================================================\n")
        raise

    sid_map = {}
    for item_id, sid_tokens in data.items():
        if sid_tokens is None:
            sid_map[item_id] = None
        elif isinstance(sid_tokens, list):
            sid_map[item_id] = "".join(str(x) for x in sid_tokens)
        else:
            sid_map[item_id] = str(sid_tokens)

    logging.info("item2sid size = %d", len(sid_map))
    return sid_map


def merge_sid_to_parquet(
        item2sid_path: str,
        input_item_info: str,
        output_parquet: str,
        show_col: list,
        primary_key: str,
        base_dir: str = "",
        sid_col: str = "sid",
        depth: int = 3,
        output_sep: str = "|",
) -> None:
    """
    执行 SID 合并到 Parquet 的主逻辑

    Args:
        item2sid_path: item2sid.json 文件路径
        input_item_info: 应用信息文件路径（可为空）
        output_parquet: 输出 parquet 文件路径
        show_col: 要输出的列列表
        primary_key: 主键列名
        base_dir: 基础工作目录
        sid_col: SID 列名
        depth: SID 解析深度
        output_sep: CSV 输出分隔符
    """
    logging.info("item2sid_path=%s", item2sid_path)

    sid_map = load_item2sid(item2sid_path)

    logging.info("input_item_info=%s, bool=%s", repr(input_item_info), bool(input_item_info))
    if input_item_info and input_item_info.strip():

        logging.info("Loading input_item_info: %s", input_item_info)
        if input_item_info.lower().endswith('.txt'):
            df = pd.read_csv(input_item_info, encoding='utf-8-sig', sep='\x01', quoting=csv.QUOTE_NONE)
        else:
            df = pd.read_csv(input_item_info, encoding="utf-8-sig", low_memory=False)
        logging.info("Item info rows = %d", len(df))

        logging.info("Merging sid by %s", primary_key)
        df[sid_col] = df[primary_key].map(sid_map)

        missing = df[sid_col].isna().sum()
        missing_ratio = missing / max(len(df), 1)
        logging.info("Missing sid count = %d / %d (%.4f)", missing, len(df), missing_ratio)

        df = df[df[sid_col].notna()].reset_index(drop=True)
        logging.info("After filtering, remaining rows = %d", len(df))

        output_cols = []
        for col in show_col:
            if col in df.columns:
                output_cols.append(col)
            else:
                df[col] = pd.NA
                output_cols.append(col)
                logging.warning("Column '%s' not found in input_item_info, using pd.NA", col)

        output_cols.append("sid")

        print("\n========== MERGED PREVIEW (TOP 10) ==========\n")
        display_cols = [c for c in output_cols if c in df.columns]
        print(df[display_cols].head(10).to_string(index=False))
        print("\n=============================================\n")

        df_output = df[output_cols]
    else:
        logging.info("No input_item_info provided, output itemid -> sid mapping only")

        item_ids = list(sid_map.keys())
        df_output = pd.DataFrame({primary_key: item_ids})
        df_output[sid_col] = df_output[primary_key].map(sid_map)

        missing = df_output[sid_col].isna().sum()
        missing_ratio = missing / max(len(df_output), 1)
        logging.info("Missing sid count = %d / %d (%.4f)", missing, len(df_output), missing_ratio)

        df_output = df_output[df_output[sid_col].notna()].reset_index(drop=True)
        logging.info("After filtering, remaining rows = %d", len(df_output))

        print("\n========== OUTPUT PREVIEW (TOP 10) ==========\n")
        print(df_output.head(10).to_string(index=False))
        print("\n=============================================\n")

    output_dir = os.path.dirname(output_parquet)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logging.info("Writing parquet -> %s", output_parquet)
    df_output.to_parquet(output_parquet, engine="pyarrow", index=False)
    csv_path = output_parquet.replace(".parquet", ".csv")
    logging.info("Writing csv -> %s", csv_path)
    df_output[sid_col] = df_output.apply(lambda row: parse_sid_to_codes_and_join(row[sid_col], depth), axis=1)
    df_output.to_csv(csv_path, sep=output_sep, index=False, encoding="utf-8-sig", header=False)
    logging.info("Done")


def main():
    parser = argparse.ArgumentParser(description="Merge App SID to Parquet")

    parser.add_argument("--config", type=str, help="配置文件路径 (YAML格式)")
    args = parser.parse_args()

    if not args.config:
        raise ValueError("必须提供 --config 参数指定配置文件")

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    depth = int(cfg.get("depth", 3))
    input_item2sid = paths.get("input_item2sid", "")
    input_item_info = paths.get("input_item_info", "")
    output_parquet = paths.get("output_parquet", "")
    base_dir = paths.get("base_dir", "")

    if base_dir:
        if input_item2sid and not os.path.isabs(input_item2sid):
            input_item2sid = os.path.join(base_dir, input_item2sid)
        if input_item_info and not os.path.isabs(input_item_info):
            input_item_info = os.path.join(base_dir, input_item_info)
        if output_parquet and not os.path.isabs(output_parquet):
            output_parquet = os.path.join(base_dir, output_parquet)

    output_config = cfg.get("output", {})
    show_col = output_config.get("show_col", [])
    sid_col = output_config.get("sid_col", "sid")
    output_sep = output_config.get("output_sep", "|")

    primary_key = cfg.get("primary_key", "app_id")

    logging_cfg = cfg.get("logging", {})
    if logging_cfg:
        log_level = logging_cfg.get("level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=logging_cfg.get("format", "%(asctime)s [%(levelname)s] %(message)s")
        )

    merge_sid_to_parquet(
        item2sid_path=input_item2sid,
        input_item_info=input_item_info,
        output_parquet=output_parquet,
        show_col=show_col,
        primary_key=primary_key,
        base_dir=base_dir,
        sid_col=sid_col,
        depth=depth,
        output_sep=output_sep,
    )


if __name__ == "__main__":
    main()
