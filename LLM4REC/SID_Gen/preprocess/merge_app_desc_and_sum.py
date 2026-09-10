# -*- coding: utf-8 -*-
"""
游戏数据合并与评论摘要预处理脚本
功能：
1. 合并游戏基础信息、描述信息和TapTap评论数据
2. 对合并后的评论数据进行清洗和预处理（限制评论数量和长度）

使用方法:
    python merge_app_and_sum.py --config configs/merge_app_and_sum.yaml
    python merge_app_and_sum.py --config configs/merge_app_and_sum.yaml --base_dir /your/custom/path
"""

import argparse
import ast
import csv
import json
import logging
import os
import re
from collections import defaultdict
from typing import List, Any, Dict

import json5
import numpy as np
import pandas as pd
from SID_Gen.utils.preprocessor import clean_text
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 默认编码
DEFAULT_ENCODING = 'utf-8-sig'

# CTYPE 类型映射
CTYPE_MAP = {
    0: "普通APK",
    1: "H5游戏",
    2: "快应用(RPK)",
    5: "运动手表应用",
    11: "授权NO-APK",
    12: "非授权NO-APK",
    13: "PC云游戏",
    17: "鸿蒙应用",
    18: "轻鸿蒙应用",
    -1: "其他",
}


# ============================================================
# 配置加载模块
# ============================================================

def load_config(config_path):
    """加载YAML配置文件"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_path(config, path_key, base_dir=None):
    """获取完整路径"""
    path_config = config.get('paths', {}).get(path_key, '')

    if isinstance(path_config, dict):
        path = path_config.get('path', '')
    else:
        path = str(path_config)

    if base_dir is None:
        base_dir = config.get('paths', {}).get('base_dir', '')

    if base_dir and not os.path.isabs(path):
        return os.path.join(base_dir, path)
    elif not base_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        return os.path.join(project_root, path)
    return path


# ============================================================
# 数据合并模块（来自merge_desc.py）
# ============================================================

def load_game_base_info(config, base_dir=None):
    """加载游戏基础信息"""
    path = get_path(config, 'input_game_base_info', base_dir)
    logger.info(f"加载游戏基础信息: {path}")

    df = pd.read_csv(path, sep='\x01', quoting=csv.QUOTE_NONE, encoding=DEFAULT_ENCODING, header=0, engine='python')
    logger.info(f"游戏基础信息加载完成，行数: {len(df)}, 列数: {len(df.columns)}")
    return df


def load_game_desc(config, base_dir=None):
    """加载游戏描述信息"""
    path = get_path(config, 'input_game_desc', base_dir)
    logger.info(f"加载游戏描述信息: {path}")

    column_names = config.get('game_desc_column_names', [])
    df = pd.read_csv(path, sep='\t', quoting=csv.QUOTE_NONE, encoding=DEFAULT_ENCODING, names=column_names,
                     engine='python')
    logger.info(f"游戏描述信息加载完成，行数: {len(df)}")
    return df


def load_valid_app_ids(config, base_dir=None):
    """加载有效APP ID列表"""
    path = get_path(config, 'input_valid_app_ids', base_dir)
    logger.info(f"加载有效APP ID列表: {path}")

    valid_ids = set(pd.read_csv(path, sep='\t', encoding='utf-8').iloc[:, 0])
    logger.info(f"有效APP ID加载完成，数量: {len(valid_ids)}")
    return valid_ids


def load_taptap_info(config, base_dir=None):
    """加载TapTap信息"""
    path = get_path(config, 'input_taptap_info', base_dir)
    logger.info(f"加载TapTap信息: {path}")

    df = pd.read_csv(path, sep='\x01', quoting=csv.QUOTE_NONE, encoding=DEFAULT_ENCODING, header=0, engine='python')
    logger.info(f"TapTap信息加载完成，行数: {len(df)}")
    return df


def safe_parse_dirty_dict_str(raw: str) -> dict:
    """自动修复常见脏数据语法并解析为 dict"""
    if not isinstance(raw, str):
        return raw

    s = raw.strip()

    # 1. 修复 ""内容"" 非法结构（常见于爬虫转义失败）
    # 匹配 "" 开头和 "" 结尾的内容，替换为单引号包裹
    s = re.sub(r'""(.*?)""', r'"\1"', s, flags=re.DOTALL)
    # 兼容其他位置的多余双引号（如未转义的 ""）
    s = re.sub(r'(?<!\\)""', '"', s)

    # 2. 剥离最外层可能存在的冗余引号

    single_noisy_flg = s.startswith("'") and s.endswith("'")
    double_noisy_flg = s.startswith('"') and s.endswith('"')
    if single_noisy_flg or double_noisy_flg:
        s = s[1:-1]

    # 3. 替换 None 为 null（json5 要求），保留 Python 原生语法给 ast
    s_py = re.sub(r'\bNone\b', 'None', s)  # ast 用
    s_json5 = re.sub(r'\bNone\b', 'null', s)  # json5 用

    # 4. 尝试解析（优先 ast，因其原生支持单引号+None）
    try:
        return ast.literal_eval(s_py)
    except (ValueError, SyntaxError):
        pass

    try:
        return json5.loads(s_json5)
    except Exception as e:
        raise ValueError(f"双引擎解析均失败: {e}\n清洗后片段: {s[:200]}...") from e


def parse_taptap_data(taptap_info_df):
    """解析TapTap数据"""
    taptap_id2content = defaultdict(dict)
    success_count = 0
    fail_count = 0
    failed_records = []

    logger.info("开始解析TapTap数据...")
    for idx, ele in tqdm(taptap_info_df.iterrows(), total=len(taptap_info_df), desc="解析数据"):
        try:
            try:
                app_ids = safe_parse_dirty_dict_str(ele['keycontent1'])
            except Exception as e:
                raise ValueError(f"app_ids 解析失败：{str(e)}") from e

            content = ele.get('content', '')

            try:
                content = safe_parse_dirty_dict_str(content)
            except Exception as e:
                raise ValueError(f"content 解析失败：{str(e)}") from e

            if not isinstance(content, dict):
                raise TypeError(f"content 解析后不是 dict 类型，而是 {type(content)}")

            try:
                hot_comments = content['hot_comments']
                mouth_comments = content['mouth_comments']
                app_name = content['app_name']
            except KeyError as e:
                raise KeyError(f"缺少必需字段：{str(e)}") from e

            for app_id in app_ids:
                taptap_id2content[app_id]['app_name'] = app_name
                taptap_id2content[app_id]['hot_comments'] = hot_comments
                taptap_id2content[app_id]['mouth_comments'] = mouth_comments

            success_count += 1

        except Exception as e:
            fail_count += 1
            failed_records.append({
                'index': idx,
                'app_ids': str(ele.get('keycontent1', 'unknown'))[:100],
                'error_type': type(e).__name__,
                'error_msg': str(e),
                'content_preview': str(ele.get('content', 'None'))[:500]
            })
            continue

    total_count = success_count + fail_count
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    fail_rate = (fail_count / total_count * 100) if total_count > 0 else 0

    logger.info("=" * 70)
    logger.info("                        数据解析完成统计")
    logger.info("=" * 70)
    logger.info(f"总处理行数：    {total_count}")
    logger.info(f"成功解析：      {success_count} 条")
    logger.info(f"失败解析：      {fail_count} 条")
    logger.info(f"成功率：        {success_rate:.2f}%")
    logger.info(f"失败率：        {fail_rate:.2f}%")
    logger.info(f"最终字典大小：  {len(taptap_id2content)} 条记录")
    logger.info("=" * 70)

    if failed_records:
        logger.warning(f"共有 {len(failed_records)} 条失败记录")
        for i, record in enumerate(failed_records[:10], 1):
            logger.warning(f"[{i}] 行号：{record['index']}, 错误：{record['error_msg']}")

    return taptap_id2content, failed_records


def merge_game_data(config, base_dir=None):
    """合并游戏数据"""
    logger.info("=" * 70)
    logger.info("开始合并游戏数据...")
    logger.info("=" * 70)

    # 1. 加载游戏基础信息
    input_data1 = load_game_base_info(config, base_dir)

    # 2. 加载有效APP ID并筛选
    valid_ids = load_valid_app_ids(config, base_dir)
    valid_input_data1 = input_data1[input_data1['app_id'].isin(valid_ids)]
    logger.info(f"有效APP筛选后行数: {len(valid_input_data1)}")

    # 2.1 增量更新：如果配置了 previous_res_file，则过滤掉已存在的 app_id
    previous_res_file = get_path(config, 'previous_res_file', base_dir)
    if previous_res_file and os.path.exists(previous_res_file):
        logger.info(f"增量更新模式：读取之前的结果文件: {previous_res_file}")
        previous_df = pd.read_csv(previous_res_file, encoding=DEFAULT_ENCODING)
        if 'app_id' in previous_df.columns:
            previous_app_ids = set(previous_df['app_id'].dropna().astype(str))
            valid_input_data1 = valid_input_data1[~valid_input_data1['app_id'].astype(str).isin(previous_app_ids)]
            logger.info(f"增量过滤后行数: {len(valid_input_data1)} (已排除 {len(previous_app_ids)} 个已存在的 app_id)")
        else:
            logger.warning(f"previous_res_file 中未找到 app_id 列，跳过增量过滤")
    elif previous_res_file:
        logger.warning(f"previous_res_file 配置了但文件不存在: {previous_res_file}，跳过增量过滤")

    # 3. 加载游戏描述信息并合并
    input_data2 = load_game_desc(config, base_dir)
    input2_subset = input_data2[['app_id', 'app_desc']]

    input1_add_desc = valid_input_data1.merge(
        input2_subset,
        on='app_id',
        how='left'
    )
    logger.info(f"合并描述后行数: {len(input1_add_desc)}")
    logger.info(f"app_desc 缺失值数量：{input1_add_desc['app_desc'].isna().sum()}")

    # 4. 加载TapTap信息并解析
    taptap_info_df = load_taptap_info(config, base_dir)
    taptap_id2content, failed_records = parse_taptap_data(taptap_info_df)

    # 5. 提取hot_comments和mouth_comments映射
    logger.info("正在提取 hot_comments 和 mouth_comments...")

    app_id2hot_comments = {
        app_id: data.get('hot_comments', [])
        for app_id, data in taptap_id2content.items()
    }

    app_id2mouth_comments = {
        app_id: data.get('mouth_comments', [])
        for app_id, data in taptap_id2content.items()
    }

    logger.info(f"hot_comments 映射表大小：{len(app_id2hot_comments)}")
    logger.info(f"mouth_comments 映射表大小：{len(app_id2mouth_comments)}")

    # 6. 添加到DataFrame
    logger.info("正在添加新列到 DataFrame...")
    input1_add_desc['hot_comments'] = input1_add_desc['app_id'].map(app_id2hot_comments)
    input1_add_desc['mouth_comments'] = input1_add_desc['app_id'].map(app_id2mouth_comments)

    # 7. 清理空列表（如果配置启用）
    if config.get('processing', {}).get('clean_empty_lists', True):
        logger.info("正在清理空列表...")

        def is_empty_list(x):
            # 检查 Python list 或 numpy array 是否为空
            if isinstance(x, (list, np.ndarray)):
                return len(x) == 0
            return False

        def replace_empty_list_with_nan(series):
            return series.apply(lambda x: np.nan if is_empty_list(x) else x)

        hot_empty_before = input1_add_desc['hot_comments'].apply(is_empty_list).sum()
        mouth_empty_before = input1_add_desc['mouth_comments'].apply(is_empty_list).sum()

        input1_add_desc['hot_comments'] = replace_empty_list_with_nan(input1_add_desc['hot_comments'])
        input1_add_desc['mouth_comments'] = replace_empty_list_with_nan(input1_add_desc['mouth_comments'])

        logger.info(f"✅ hot_comments: 替换了 {hot_empty_before} 个空列表")
        logger.info(f"✅ mouth_comments: 替换了 {mouth_empty_before} 个空列表")

    # 8. 统计映射情况
    total_rows = len(input1_add_desc)
    hot_comments_has_data = input1_add_desc['hot_comments'].notna().sum()
    mouth_comments_has_data = input1_add_desc['mouth_comments'].notna().sum()
    any_has_data = (input1_add_desc['hot_comments'].notna() | input1_add_desc['mouth_comments'].notna()).sum()
    completely_nan = (input1_add_desc['hot_comments'].isna() & input1_add_desc['mouth_comments'].isna()).sum()

    logger.info("=" * 70)
    logger.info("                        映射统计结果")
    logger.info("=" * 70)
    logger.info(f"input1_add_desc 总行数：              {total_rows}")
    logger.info(f"input1_add_desc 唯一 app_id 数：       {input1_add_desc['app_id'].nunique()}")
    logger.info(
        f"hot_comments 匹配成功：{hot_comments_has_data} 行 ({hot_comments_has_data / total_rows * 100:.2f}%)")
    logger.info(
        f"mouth_comments 匹配成功：{mouth_comments_has_data} 行 ({mouth_comments_has_data / total_rows * 100:.2f}%)")
    logger.info(f"至少一个字段有实际数据：{any_has_data} 行 ({any_has_data / total_rows * 100:.2f}%)")
    logger.info(
        f"完全未匹配：{completely_nan} 行 ({100 - any_has_data / total_rows * 100:.2f}%)")
    logger.info("=" * 70)

    return input1_add_desc, failed_records, taptap_info_df


def parse_json_list(comment_str: str) -> List[Dict]:
    """解析 JSON 列表字符串"""
    if not comment_str or str(comment_str).strip() == "":
        return []

    comment_str = str(comment_str).strip()

    try:
        return json5.loads(comment_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(comment_str)
        except (ValueError, SyntaxError):
            logger.warning(f"JSON解析失败: {comment_str[:100]}...")
            return []


def extract_content_from_json_list(json_list: List[Dict], json_key: str = "content") -> List[str]:
    """从 JSON 列表中提取指定字段"""
    contents = []
    for item in json_list:
        if isinstance(item, dict) and json_key in item:
            content = item[json_key]
            if content and str(content).strip():
                contents.append(str(content).strip())
    return contents


def merge_comments(
        hot_comments: Any,
        mouth_comments: Any,
        max_items: int = 20,
        max_item_length: int = 2000,
        json_key: str = "content"
) -> List[str]:
    """合并 hot_comments 和 mouth_comments"""
    all_comments = []

    # 解析 hot_comments - 使用 pd.notna() 处理 numpy 数组
    if isinstance(hot_comments, list) and len(hot_comments) > 0:
        hot_list = parse_json_list(str(hot_comments))
        hot_contents = extract_content_from_json_list(hot_list, json_key)
        all_comments.extend(hot_contents)

    # 解析 mouth_comments - 使用 pd.notna() 处理 numpy 数组
    if isinstance(mouth_comments, list) and len(mouth_comments) > 0:
        mouth_list = parse_json_list(str(mouth_comments))
        mouth_contents = extract_content_from_json_list(mouth_list, json_key)
        all_comments.extend(mouth_contents)  # 限制数量
    if len(all_comments) > max_items:
        all_comments = all_comments[:max_items]

    # 限制每条长度
    truncated_comments = []
    for comment in all_comments:
        if len(comment) > max_item_length:
            truncated_comments.append(comment[:max_item_length] + "...")
        else:
            truncated_comments.append(comment)

    return truncated_comments


def preprocess_comments(df, config):
    """预处理评论数据

    Args:
        df: 包含评论数据的DataFrame
        config: 配置字典

    Returns:
        pd.DataFrame: 预处理后的DataFrame
    """
    # 获取预处理配置
    preprocess_config = config.get('preprocess', {})

    max_items = preprocess_config.get('max_items', 20)
    max_item_length = preprocess_config.get('max_item_length', 2000)
    json_key = preprocess_config.get('json_key', 'content')
    output_col = preprocess_config.get('output_col', 'comments')

    hot_comments_col = preprocess_config.get('hot_comments_col', 'hot_comments')
    mouth_comments_col = preprocess_config.get('mouth_comments_col', 'mouth_comments')

    logger.info("=" * 70)
    logger.info("开始预处理评论数据...")
    logger.info("=" * 70)
    logger.info(f"max_items: {max_items}")
    logger.info(f"max_item_length: {max_item_length}")
    logger.info(f"json_key: {json_key}")
    logger.info(f"output_col: {output_col}")

    # 预处理每行数据
    comments_list = []
    valid_count = 0
    empty_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="预处理评论"):
        merged = merge_comments(
            row[hot_comments_col],
            row[mouth_comments_col],
            max_items=max_items,
            max_item_length=max_item_length,
            json_key=json_key
        )

        if merged:
            valid_count += 1
            # 将评论列表转换为换行符分隔的字符串
            comments_str = "\n".join(merged)
            comments_str = clean_text(comments_str, keep_punctuation=True)
        else:
            empty_count += 1
            # 空列表写入空字符串
            comments_str = ""

        comments_list.append(comments_str)

    # 添加输出列
    df[output_col] = comments_list

    logger.info("=" * 70)
    logger.info("                        评论预处理完成统计")
    logger.info("=" * 70)
    logger.info(f"总数据量：    {len(df)} 条")
    df_len = len(df)
    valid_pct = valid_count / df_len * 100 if df_len > 0 else 0
    empty_pct = empty_count / df_len * 100 if df_len > 0 else 0
    logger.info(f"有效数据：    {valid_count} 条 ({valid_pct:.2f}%)")
    logger.info(f"空数据：      {empty_count} 条 ({empty_pct:.2f}%)")
    logger.info("=" * 70)

    return df


# ============================================================
# 主流程
# ============================================================

def run_pipeline(config, base_dir=None):
    """运行完整的处理流程

    Args:
        config: 配置字典
        base_dir: 基础目录
    """
    logger.info("=" * 70)
    logger.info("           游戏数据合并与评论预处理流程启动")
    logger.info("=" * 70)

    # 步骤1: 合并游戏数据
    df, failed_records, taptap_info_df = merge_game_data(config, base_dir)

    # 步骤2: 获取输出目录
    output_merged_data_path = get_path(config, 'output_merged_data', base_dir)
    output_dir = os.path.dirname(output_merged_data_path)
    os.makedirs(output_dir, exist_ok=True)

    # 步骤3: 保存未匹配的app_id到输出目录
    completely_nan = (df['hot_comments'].isna() & df['mouth_comments'].isna()).sum()
    if completely_nan > 0:
        unmatched_df = df.loc[
            df['hot_comments'].isna() & df['mouth_comments'].isna(),
            ['app_id']
        ].drop_duplicates()
        unmatched_path = os.path.join(output_dir, 'unmatched_app_ids.csv')
        unmatched_df.to_csv(unmatched_path, index=False, encoding=DEFAULT_ENCODING)
        logger.info(f"✅ 未匹配的 app_id 已保存到：{unmatched_path}")

    # 步骤4: 保存映射统计到输出目录
    total_rows = len(df)
    hot_comments_has_data = df['hot_comments'].notna().sum()
    mouth_comments_has_data = df['mouth_comments'].notna().sum()
    any_has_data = (df['hot_comments'].notna() | df['mouth_comments'].notna()).sum()

    summary_df = pd.DataFrame([{
        '总行数': total_rows,
        '唯一 app_id 数': df['app_id'].nunique(),
        'hot_comments_有数据': hot_comments_has_data,
        'hot_comments_NaN': df['hot_comments'].isna().sum(),
        'mouth_comments_有数据': mouth_comments_has_data,
        'mouth_comments_NaN': df['mouth_comments'].isna().sum(),
        '至少一个有数据': any_has_data,
        '两个都 NaN': completely_nan,
        '有数据匹配率': f"{any_has_data / total_rows * 100:.2f}%" if total_rows > 0 else "0.00%"
    }])
    stats_path = os.path.join(output_dir, 'mapping_statistics.csv')
    summary_df.to_csv(stats_path, index=False, encoding=DEFAULT_ENCODING)
    logger.info(f"✅ 统计结果已保存到：{stats_path}")

    # 步骤5: 保存解析失败记录到输出目录
    if failed_records:
        failed_full_records = []
        for record in failed_records:
            idx = record['index']
            if idx < len(taptap_info_df):
                full_content = str(taptap_info_df.iloc[idx]['content'])
            else:
                full_content = 'N/A'
            failed_full_records.append({
                'index': record['index'],
                'app_ids': record['app_ids'],
                'error_type': record['error_type'],
                'error_msg': record['error_msg'],
                'raw_content': full_content
            })

        failed_full_df = pd.DataFrame(failed_full_records)
        failed_path = os.path.join(output_dir, 'parse_failed_records_full.csv')
        failed_full_df.to_csv(failed_path, index=False, encoding=DEFAULT_ENCODING)
        logger.info(f"解析失败记录已保存到：{failed_path}")

    # 步骤6: 预处理评论数据
    df = preprocess_comments(df, config)

    # 步骤7: 转换 ctype 列
    if 'ctype' in df.columns:
        logger.info("正在转换 ctype 列...")
        original_ctype_count = df['ctype'].notna().sum()

        def convert_ctype(v):
            """将 ctype 的 float 值转换为对应的中文描述"""
            if pd.isna(v) or v == '' or v is None:
                return "其他"
            try:
                int_value = int(float(v))
                return CTYPE_MAP.get(int_value, "其他")
            except (ValueError, TypeError):
                return "其他"

        df['ctype'] = df['ctype'].apply(convert_ctype)
        logger.info(f"ctype 列转换完成，有效值数量: {original_ctype_count}")

    # 步骤8: 保存合并并预处理后的数据
    df.to_csv(output_merged_data_path, index=False, encoding=DEFAULT_ENCODING)
    logger.info(f"合并并预处理后的数据已保存到：{output_merged_data_path}")

    logger.info("=" * 70)
    logger.info("           游戏数据合并与评论预处理流程完成！")
    logger.info("=" * 70)

    return df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='游戏数据合并与评论预处理脚本')

    # 配置文件
    parser.add_argument('--config', type=str, default='configs/merge_app_and_sum.yaml',
                        help='配置文件路径（默认：configs/merge_app_and_sum.yaml）')

    # 基础目录（覆盖配置文件中的base_dir）
    parser.add_argument('--base_dir', type=str, default=None,
                        help='基础工作目录（如果指定，则优先使用）')

    # 调试模式
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')

    args = parser.parse_args()

    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("调试模式已启用")

    # 加载配置
    if not os.path.isabs(args.config):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(script_dir), args.config)
    else:
        config_path = args.config

    logger.info(f"加载配置文件: {config_path}")
    config = load_config(config_path)

    # 运行流程
    try:
        run_pipeline(config, args.base_dir)
        logger.info("任务执行成功！")
    except Exception as e:
        logger.error(f"任务执行失败：{str(e)}")
        raise


if __name__ == '__main__':
    main()
