# -*- coding: utf-8 -*-
"""
CSV表头添加脚本
为CSV文件添加自定义表头

使用示例：
    python preprocess/add_csv_header.py --config configs/add_csv_header.yaml
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.log_utils import get_logger

QUOTING_MAP = {
    "minimal": csv.QUOTE_MINIMAL,
    "none": csv.QUOTE_NONE,
    "all": csv.QUOTE_ALL,
    "nonnumeric": csv.QUOTE_NONNUMERIC,
}


def parse_args():
    parser = argparse.ArgumentParser(description="CSV表头添加器")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--input_file", type=str, default=None, help="输入CSV文件")
    parser.add_argument("--output_file", type=str, default=None, help="输出CSV文件")
    parser.add_argument("--header", type=str, default=None, help="表头，逗号分隔")
    parser.add_argument("--sep", type=str, default=None, help="CSV分隔符")
    parser.add_argument("--quoting", type=str, default=None, help="CSV引号模式")
    return parser.parse_args()


def merge_config_with_args(config: Dict, args) -> argparse.Namespace:
    merged = argparse.Namespace()
    merged.input_file = args.input_file if args.input_file else config.get("input_file", "")
    merged.output_file = args.output_file if args.output_file else config.get("output_file", "")
    merged.header = args.header if args.header else config.get("header", [])
    merged.sep = args.sep if args.sep else config.get("sep", ",")
    merged.quoting = args.quoting if args.quoting else config.get("quoting", "minimal")
    merged.log_level = config.get("log_level", "INFO")
    merged._config_path = args.config
    return merged


def validate_args(args):
    if not args._config_path:
        raise ValueError("配置文件路径不能为空")
    if not os.path.exists(args._config_path):
        raise ValueError(f"配置文件不存在: {args._config_path}")
    if not args.input_file:
        raise ValueError("输入文件路径不能为空")
    if not args.output_file:
        raise ValueError("输出文件路径不能为空")
    if not args.header:
        raise ValueError("表头不能为空")
    return args


def add_header_to_csv(
        input_file: str,
        output_file: str,
        header: List[str],
        logger: logging.Logger,
        sep: str,
        quoting: str = "minimal",
) -> int:
    quoting_value = QUOTING_MAP.get(quoting, csv.QUOTE_MINIMAL)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logger.info("Loading CSV: %s", input_file)
    df = pd.read_csv(input_file, encoding='utf-8-sig', sep=sep, header=None, quoting=quoting_value)
    original_count = len(df)
    logger.info("Original rows: %d", original_count)

    if len(header) != len(df.columns):
        raise ValueError(f"Header length ({len(header)}) does not match CSV columns ({len(df.columns)})")

    df.columns = header

    output_dir = Path(output_file).parent
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_file, index=False, encoding="utf-8-sig", sep=sep, quoting=quoting_value)
    logger.info("Saved to: %s", output_file)

    return original_count


def main():
    args = parse_args()
    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    args = merge_config_with_args(config, args)
    args = validate_args(args)

    log_dir = os.path.dirname(args.output_file) if args.output_file else None
    logger = get_logger("add_csv_header", log_dir=log_dir)

    result_count = add_header_to_csv(
        input_file=args.input_file,
        output_file=args.output_file,
        header=args.header,
        logger=logger,
        sep=args.sep,
        quoting=args.quoting,
    )

    logger.info("Add header completed. Output rows: %d", result_count)


if __name__ == "__main__":
    main()
