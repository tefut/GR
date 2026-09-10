# -*- coding: utf-8 -*-
"""
Embedding拼接预处理脚本
功能：
1. 加载npz文件中的embedding数据
2. 加载csv/txt文件中的embedding数据
3. 根据primary_key进行映射拼接
4. 支持l2 norm配置

使用方法:
    python merge_embeddings.py --config configs/merge_embeddings.yaml
    python merge_embeddings.py --config configs/merge_embeddings.yaml --base_dir /your/custom/path
"""

import argparse
import csv
import logging
import os

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_ENCODING = 'utf-8-sig'


def load_config(config_path):
    """加载YAML配置文件"""
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


def load_npz_data(config, base_dir=None):
    """加载npz文件中的embedding数据"""
    path = get_path(config, 'input_npz', base_dir)
    logger.info(f"加载npz文件: {path}")

    npz_data = np.load(path, allow_pickle=True)
    id_key = config.get('npz', {}).get('id_key', 'id')
    embedding_key = config.get('npz', {}).get('embedding_key', 'embedding')

    if id_key not in npz_data:
        raise KeyError(f"npz文件中不存在的id_key: {id_key}")
    if embedding_key not in npz_data:
        raise KeyError(f"npz文件中不存在的embedding_key: {embedding_key}")

    ids = npz_data[id_key]
    embeddings = npz_data[embedding_key]

    logger.info(
        f"npz数据加载完成，id数量: {len(ids)}, embedding维度: {embeddings.shape[1] if len(embeddings.shape) > 1 else 1}")

    return ids, embeddings, id_key


def load_csv_data(config, base_dir=None):
    """加载csv/txt文件中的embedding数据"""
    path = get_path(config, 'input_csv', base_dir)
    logger.info(f"加载csv/txt文件: {path}")

    embedding_columns = config.get('csv', {}).get('embedding_columns', [])
    id_column = config.get('csv', {}).get('id_column', 'id')
    sep = config.get('csv', {}).get('separator', ',')

    usecols = [id_column] + embedding_columns
    if path.endswith('.txt'):
        df = pd.read_csv(path, sep=sep, encoding=DEFAULT_ENCODING, quoting=csv.QUOTE_NONE, header=0, engine='python',
                         usecols=usecols)
    else:
        df = pd.read_csv(path, encoding=DEFAULT_ENCODING, header=0, engine='python', usecols=usecols)

    logger.info(f"csv数据加载完成，行数: {len(df)}, 列数: {len(df.columns)}")

    if id_column not in df.columns:
        raise KeyError(f"csv文件中不存在的id列: {id_column}")
    logger.info(f"csv id列名: {id_column}")

    for col in embedding_columns:
        if col not in df.columns:
            raise KeyError(f"csv文件中不存在的embedding列: {col}")
        logger.info(f"  embedding列 '{col}' 存在")

    return df, embedding_columns, id_column


def parse_embedding_string(emb_str):
    """解析逗号分隔的embedding字符串为numpy数组"""
    if pd.isna(emb_str):
        return None
    try:
        return np.array([float(x) for x in str(emb_str).split(',')])
    except ValueError:
        logger.warning(f"无法解析embedding字符串: {emb_str[:50]}...")
        return None


def l2_normalize(embeddings):
    """对embedding进行l2 normalization"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def merge_embeddings(config, base_dir=None):
    """合并npz和csv中的embedding数据"""
    logger.info("=" * 70)
    logger.info("开始合并embedding数据...")
    logger.info("=" * 70)

    npz_ids, npz_embeddings, npz_id_key = load_npz_data(config, base_dir)
    csv_df, csv_embedding_columns, csv_id_column = load_csv_data(config, base_dir)

    norm_before_concat = config.get('processing', {}).get('norm_before_concat', False)
    logger.info(f"norm_before_concat: {norm_before_concat}")

    id_to_npz_idx = {id_val: idx for idx, id_val in enumerate(npz_ids)}

    npz_embedding_dim = npz_embeddings.shape[1] if len(npz_embeddings.shape) > 1 else 1
    csv_embedding_dims = {}
    sample_csv_embedding = None
    for col in csv_embedding_columns:
        sample_val = csv_df[col].dropna().iloc[0] if len(csv_df[col].dropna()) > 0 else None
        if sample_val is not None:
            parsed = parse_embedding_string(sample_val)
            if parsed is not None:
                csv_embedding_dims[col] = len(parsed)

    logger.info(f"npz embedding维度: {npz_embedding_dim}")
    logger.info(f"csv embedding维度: {csv_embedding_dims}")

    merged_embeddings = []
    merged_ids = []
    match_count = 0
    unmatch_count = 0

    for _, row in csv_df.iterrows():
        csv_id = row[csv_id_column]
        if csv_id in id_to_npz_idx:
            match_count += 1
            npz_idx = id_to_npz_idx[csv_id]
            npz_emb = npz_embeddings[npz_idx].flatten()

            if norm_before_concat:
                npz_emb = l2_normalize(npz_emb.reshape(1, -1)).flatten()

            csv_emb_parts = []
            for col in csv_embedding_columns:
                parsed = parse_embedding_string(row[col])
                if parsed is not None:
                    if norm_before_concat:
                        parsed = l2_normalize(parsed.reshape(1, -1)).flatten()
                    csv_emb_parts.append(parsed)
                else:
                    csv_emb_parts.append(np.zeros(csv_embedding_dims.get(col, npz_embedding_dim)))

            final_emb = np.concatenate([npz_emb] + csv_emb_parts)
            merged_embeddings.append(final_emb)
            merged_ids.append(csv_id)
        else:
            unmatch_count += 1

    logger.info(f"匹配成功: {match_count} 条")
    logger.info(f"未匹配: {unmatch_count} 条")

    if merged_embeddings:
        merged_embeddings = np.array(merged_embeddings)
        logger.info(f"合并后embedding维度: {merged_embeddings.shape[1]}")

        return merged_ids, merged_embeddings, match_count, unmatch_count
    else:
        logger.warning("没有找到任何匹配的记录")
        return None, None, match_count, unmatch_count


def run_pipeline(config, base_dir=None):
    """运行完整的处理流程"""
    logger.info("=" * 70)
    logger.info("           Embedding拼接预处理流程启动")
    logger.info("=" * 70)

    merged_ids, merged_embeddings, match_count, unmatch_count = merge_embeddings(config, base_dir)

    if merged_ids is None:
        logger.error("没有可保存的数据")
        return None, None

    output_path = get_path(config, 'output_npz', base_dir)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    output_id_key = config.get('output', {}).get('id_key', 'id')
    output_embedding_key = config.get('output', {}).get('embedding_key', 'embedding')

    np.savez(output_path, **{output_id_key: np.array(merged_ids), output_embedding_key: merged_embeddings})
    logger.info(f"合并后的数据已保存到: {output_path}")

    logger.info("=" * 70)
    logger.info("                        处理完成统计")
    logger.info("=" * 70)
    logger.info(f"总行数: {len(merged_ids)}")
    logger.info(f"匹配成功: {match_count}")
    logger.info(f"未匹配: {unmatch_count}")
    logger.info("=" * 70)

    logger.info("           Embedding拼接预处理流程完成！")
    logger.info("=" * 70)

    return merged_ids, merged_embeddings


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Embedding拼接预处理脚本')

    parser.add_argument('--config', type=str, default='configs/merge_embeddings.yaml',
                        help='配置文件路径（默认: configs/merge_embeddings.yaml）')

    parser.add_argument('--base_dir', type=str, default=None,
                        help='基础工作目录（如果指定，则优先使用）')

    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("调试模式已启用")

    if not os.path.isabs(args.config):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(script_dir), args.config)
    else:
        config_path = args.config

    logger.info(f"加载配置文件: {config_path}")
    config = load_config(config_path)

    try:
        run_pipeline(config, args.base_dir)
        logger.info("任务执行成功！")
    except Exception as e:
        logger.error(f"任务执行失败: {str(e)}")
        raise


if __name__ == '__main__':
    main()
