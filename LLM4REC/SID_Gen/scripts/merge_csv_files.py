# -*- coding: utf-8 -*-
"""
CSV文件合并脚本
根据 primary_key 合并多个 CSV 文件

功能：
- 读取多个 CSV 文件第一个文件为主）
- 每个文件的 input_column 可以不同
- 根据 primary_key 进行合并，保留每个文件的原始列名

使用示例：
    python scripts/merge_csv_files.py --config configs/merge_csv.yaml

    # 直接定参数
    python scripts/merge_csv_files.py \
        --input_files file1.csv,file2.csv,file3.csv \
        --primary_key app_id \
        --input_columns desc,summary,comment \
        --output_file merged.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# 添加项目根目录到路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.log_utils import get_logger


# ============================================
# 配置加载
# ============================================

def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置"""
    import yaml

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


# ============================================
# Union 逻辑
# ============================================

def union_csv_files(
        input_files: List[str],
        output_file: str,
        primary_key: str,
        input_columns: List[str],
        logger: logging.Logger,
) -> pd.DataFrame:
    """
    Union 多个 CSV 文件（类似 SQL UNION）
    支持增量更新：根据 primary_key 去重，保留最后出现的数据

    Args:
        input_files: CSV 文件路径列表
        output_file: 输出文件路径
        primary_key: 主键列名
        input_columns: 此参数在 union 模式下被忽略（文件需具有相同的列结构）
        logger: 日志器

    Returns:
        pd.DataFrame: 合并后的 DataFrame
    """
    all_dfs = []

    for csv_file in input_files:
        logger.info("Loading file: %s", csv_file)
        df_current = pd.read_csv(csv_file)

        if primary_key not in df_current.columns:
            logger.warning("Primary key '%s' not found in %s, skipping", primary_key, csv_file)
            continue

        logger.info("  rows: %d, columns: %s", len(df_current), list(df_current.columns))
        all_dfs.append(df_current)

    if not all_dfs:
        logger.warning("No valid data to merge")
        df_result = pd.DataFrame(columns=[primary_key])
    else:
        # 纵向拼接
        df_combined = pd.concat(all_dfs, ignore_index=True)
        logger.info("Combined rows before dedup: %d", len(df_combined))

        # 按 primary_key 去重，保留最后出现的数据（增量更新）
        df_result = df_combined.drop_duplicates(subset=[primary_key], keep="last")
        logger.info("After dedup rows: %d", len(df_result))

    # 保存结果
    output_dir = Path(output_file).parent
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if len(df_result) == 0:
        df_empty = pd.DataFrame(columns=df_result.columns if len(df_result.columns) > 0 else [primary_key])
        df_empty.to_csv(output_file, index=False, encoding="utf-8")
        logger.info("Saved empty DataFrame with headers to: %s", output_file)
    else:
        df_result.to_csv(output_file, index=False, encoding="utf-8")
        logger.info("Saved merged data to: %s", output_file)

    return df_result


# ============================================
# 合并逻辑
# ============================================

def merge_csv_files(
        input_files: List[str],
        output_file: str,
        primary_key: str,
        input_columns: List[str],
        logger: logging.Logger,
        how: str = "left",
        mode: str = "join",
) -> pd.DataFrame:
    """
    合并多个 CSV 文件

    Args:
        input_files: CSV 文件路径列表
        output_file: 输出文件路径
        primary_key: 主键列名
        input_columns: 各文件要合并的列名列表（与 input_files 一一对应）
        logger: 日志器
        how: 合并方式 ('left', 'right', 'outer', 'inner')

    Returns:
        pd.DataFrame: 合并后的 DataFrame
    """
    if not input_files:
        raise ValueError("input_files cannot be empty")

    # 检查所有文件是否存在
    for f in input_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Input file not found: {f}")

    logger.info("Merging %d CSV files, mode: %s", len(input_files), mode)

    # Union 模式：类似 SQL UNION，支持增量更新
    if mode == "union":
        return union_csv_files(input_files, output_file, primary_key, input_columns, logger)

    # Join 模式需要验证 input_columns 数量
    if len(input_files) != len(input_columns):
        raise ValueError(f"input_files ({len(input_files)}) and input_columns ({len(input_columns)}) count mismatch")

    # Join 模式：原有逻辑
    logger.info("Primary key: %s", primary_key)
    logger.info("Input columns: %s", input_columns)

    # 读取第一个文件作为基
    base_file = input_files[0]
    base_col = input_columns[0]
    logger.info("Loading base file: %s", base_file)
    logger.info("Base input column: %s", base_col)

    df_merged = pd.read_csv(base_file)

    # 验必需列
    if primary_key not in df_merged.columns:
        raise ValueError(f"Primary key '{primary_key}' not found in {base_file}")

    if base_col == "_ALL_":
        logger.info("Using all columns from base file")
        df_base = df_merged.copy()
    else:
        if base_col not in df_merged.columns:
            raise ValueError(f"Input column '{base_col}' not found in {base_file}")
        df_base = df_merged[[primary_key, base_col]].copy()

    logger.info("Base file rows: %d", len(df_merged))

    # 遍历剩余文件行合并
    for i, (csv_file, input_col) in enumerate(zip(input_files[1:], input_columns[1:]), start=1):
        logger.info("Merging file %d: %s (column: %s)", i, csv_file, input_col)

        df_current = pd.read_csv(csv_file)

        # 验证必需列
        if primary_key not in df_current.columns:
            logger.warning("Primary key '%s' not found in %s, skipping", primary_key, csv_file)
            continue

        if input_col == "_ALL_":
            logger.info("Using all columns from %s", csv_file)
            cols_to_merge = [col for col in df_current.columns if col != primary_key]
        else:
            if input_col not in df_current.columns:
                logger.warning("Input column '%s' not found in %s, skipping", input_col, csv_file)
                continue
            cols_to_merge = [input_col]

        # 找出同名列（除primary_key外），用merge的值覆盖base的值
        common_cols = set(df_base.columns) & set(cols_to_merge)
        if common_cols:
            logger.info("Overlapping columns found: %s, will be overwritten", list(common_cols))

        # 以primary_key为索引进行合并覆盖
        df_current_to_merge = df_current[[primary_key] + cols_to_merge].set_index(primary_key)

        for col in cols_to_merge:
            if col in df_base.columns:
                # 同名列：用merge的值覆盖
                mask = df_base[primary_key].isin(df_current[primary_key])
                df_base.loc[mask, col] = df_base.loc[mask, primary_key].map(
                    df_current_to_merge[col]
                ).fillna(df_base.loc[mask, col])
            else:
                # 新列：直接添加
                df_merged = df_base.merge(
                    df_current[[primary_key, col]],
                    on=primary_key,
                    how="left"
                )
                df_base = df_merged

        logger.info("After merge rows: %d", len(df_base))

    # 保存结果
    output_dir = Path(output_file).parent
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if len(df_base) == 0:
        df_empty = pd.DataFrame(columns=df_base.columns)
        df_empty.to_csv(output_file, index=False, encoding="utf-8")
        logger.info("Saved empty DataFrame with headers to: %s", output_file)
    else:
        df_base.to_csv(output_file, index=False, encoding="utf-8")
        logger.info("Saved merged data to: %s", output_file)

    return df_base


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CSV文件合并 - 根据primary_key合并多个CSV的input_column",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用配置文件
  python scripts/merge_csv_files.py --config configs/merge_csv.yaml

  # 直接指定参数（逗号分隔）
  python scripts/merge_csv_files.py \\
      --input_files file1.csv,file2.csv,file3.csv \\
      --primary_key app_id \\
      --input_columns desc,summary,comment \\
      --output_file merged.csv
        """
    )

    parser.add_argument("--config", type=str, default=None, help="配置文件路径")

    # 文件路径参数
    parser.add_argument("--input_files", nargs='+', default=None, help="输入CSV文件列表")
    parser.add_argument("--output_file", type=str, default=None, help="输出CSV文件路径")

    # 合并参数
    parser.add_argument("--primary_key", type=str, default=None, help="主键列名（所有文件相同）")
    parser.add_argument("--input_columns", nargs='+', default=None, help="各文件要合并的列名")

    # 额外参
    parser.add_argument("--how", type=str, default=None, choices=["left", "right", "outer", "inner"],
                        help="合并方式 (default: left)")
    parser.add_argument("--mode", type=str, default=None, choices=["join", "union"],
                        help="合并模式: join=根据primary_key关联, union=纵向拼接去重 (default: join)")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别")

    return parser.parse_args()


def merge_config_with_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """合并配置和命令行参数，命令行参数优先"""
    merged = {}

    merged["input_files"] = args.input_files if args.input_files else config.get("input_files", [])
    merged["output_file"] = args.output_file if args.output_file else config.get("output_file", "")
    merged["primary_key"] = args.primary_key if args.primary_key else config.get("primary_key", "")
    merged["input_columns"] = args.input_columns if args.input_columns else config.get("input_columns", [])
    merged["how"] = args.how if args.how else config.get("how", "left")
    merged["mode"] = args.mode if args.mode else config.get("mode", "join")
    merged["log_level"] = args.log_level if args.log_level else config.get("log_level", "INFO")

    return merged


def main():
    """主函数"""
    args = parse_args()

    # 如果提供了配置文件，加载配置
    config = {}
    if args.config:
        config = load_config(args.config)

    # 合并配置和参数
    params = merge_config_with_args(config, args)

    # 设置日志
    logger = get_logger(
        "merge_csv_files",
        log_dir=os.path.dirname(params["output_file"]) if params.get("output_file") else None
    )
    logger.info("Merge CSV Files Task Started")
    logger.info("DEBUG: mode=%s, input_files=%d, input_columns=%d",
                params.get("mode"), len(params.get("input_files", [])), len(params.get("input_columns", [])))

    # 验证必填参数
    if not params.get("input_files"):
        raise ValueError("--input_files is required (or input_files/input_files_str in config)")
    if not params.get("output_file"):
        raise ValueError("--output_file is required")
    if not params.get("primary_key"):
        raise ValueError("--primary_key is required")

    # 验证数量匹配（union 模式下忽略 input_columns）
    mode = params.get("mode", "join")
    if mode != "union":
        if not params.get("input_columns"):
            raise ValueError("--input_columns is required")
        if len(params["input_files"]) != len(params["input_columns"]):
            raise ValueError(
                f"input_files ({len(params['input_files'])}) '''"
                f"and input_columns ({len(params['input_columns'])}) count mismatch"
            )

    # 执行合并
    result_df = merge_csv_files(
        input_files=params["input_files"],
        output_file=params["output_file"],
        primary_key=params["primary_key"],
        input_columns=params["input_columns"],
        logger=logger,
        how=params.get("how", "left"),
        mode=params.get("mode", "join"),
    )

    logger.info("Merge completed. Output rows: %d, Output columns: %d",
                len(result_df), len(result_df.columns))
    logger.info("Output columns: %s", list(result_df.columns))


if __name__ == "__main__":
    main()
