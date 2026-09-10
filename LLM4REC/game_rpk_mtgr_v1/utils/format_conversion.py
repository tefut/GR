# -*- coding: utf-8 -*-
"""SID Format Conversion on MTP production environment.
将 SID 数据和 Feature 数据转换为指定格式的输出文件。
"""
import argparse
import json
import logging
import os
import re
import stat
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)

# 默认文件路径
DEFAULT_SID_PATH = "item2sid.json"
DEFAULT_FEATURE_PATH = "feature_map.json"
DEFAULT_OUTPUT_FILE = "appid2sids.txt"
# 正则表达式：匹配下划线后的数字
TAG_NUMBER_PATTERN = re.compile(r'_(\d+)>')
# 默认未匹配时的数字列表
DEFAULT_NUMERIC_LIST = [0, 0, 0]


def extract_numbers(tags: List[str]) -> List[int]:
    """从标签列表中提取数字。
    Args:
        tags: 标签列表
    Returns:
        提取的数字列表
    """
    nums = []
    for tag in tags:
        match = TAG_NUMBER_PATTERN.search(tag)
        if match:
            nums.append(int(match.group(1)))
    return nums


def load_json_file(file_path: str) -> Dict[str, Any]:
    """加载 JSON 文件。
    Args:
        file_path: 文件路径
    Returns:
        解析后的 JSON 数据
    Raises:
        FileNotFoundError: 当文件不存在时抛出
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")

    read_flags = os.O_RDONLY
    modes = stat.S_IRUSR
    with os.fdopen(os.open(file_path, read_flags, modes), 'r', encoding='UTF-8') as f:
        return json.load(f)


def process_files(sid_path: str, feature_path: str, output_path: str) -> None:
    """处理 SID 和 Feature 文件，生成转换后的输出文件。
    Args:
        sid_path: SID 数据文件路径
        feature_path: Feature 数据文件路径
        output_path: 输出文件路径
    """
    # 1. 读取 sid 文件
    sid_data = load_json_file(sid_path)
    # 2. 读取 feature 文件
    feature_data = load_json_file(feature_path)
    app_ids = feature_data.get("sparse", {}).get("app_id", {})

    unmatched_count = 0
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    modes = stat.S_IRUSR | stat.S_IWUSR
    with os.fdopen(os.open(output_path, flags, modes), 'w', encoding="utf-8") as f:
        # 3. 遍历 app_ids 进行匹配
        for key, prefix_val in app_ids.items():
            # 提取数字列表
            if sid_data.get(key):
                tag_list = sid_data[key]
                numeric_list = [x + 1 for x in extract_numbers(tag_list)]
            else:
                unmatched_count += 1
                numeric_list = DEFAULT_NUMERIC_LIST.copy()

            # 按照格式拼接：976|[114, 115, 186]
            f.write(f"{prefix_val}|{numeric_list}\n")

    logging.info(f"处理完成，共生成 {len(app_ids)} 条数据，未匹配数据：{unmatched_count}，已保存至: {output_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。
    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="SID Format Conversion on MTP production environment",
    )
    parser.add_argument(
        "--sid_path",
        type=str,
        default=DEFAULT_SID_PATH,
        help="SID 数据文件路径 (JSON 格式)",
    )
    parser.add_argument(
        "--feature_path",
        type=str,
        default=DEFAULT_FEATURE_PATH,
        help="Feature 数据文件路径 (JSON 格式)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="输出文件路径",
    )
    return parser.parse_args()


def main() -> None:
    """主函数。"""
    args = parse_args()
    process_files(args.sid_path, args.feature_path, args.output_file)


if __name__ == '__main__':
    main()
