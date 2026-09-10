#!/usr/bin/env python3
# Embedding相似度计算任务脚本

import argparse
import csv
import os

import faiss
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

try:
    import faiss

    print("Faiss imported successfully. Using Faiss for similarity computation.")
except ImportError as e:
    print(f"Failed to import faiss: {e}")
    print("Falling back to alternative similarity computation method (e.g., sklearn/numpy).")


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_embedding(npz_path, id_key='app_id', emb_key='embedding'):
    data = np.load(npz_path, allow_pickle=True)
    app_ids = data[id_key]
    embeddings = data[emb_key]
    return app_ids, embeddings


def load_name_mapping(csv_path, primary_key, name_column):
    if csv_path.lower().endswith('.txt'):
        df = pd.read_csv(csv_path, usecols=[primary_key, name_column], encoding='utf-8-sig', sep='\x01',
                         quoting=csv.QUOTE_NONE)
    else:
        df = pd.read_csv(csv_path, usecols=[primary_key, name_column])

    df = df.dropna(subset=[primary_key, name_column])
    name_mapping = dict(zip(df[primary_key].astype(str), df[name_column]))
    return name_mapping


def load_app_rank(app_rank_path, app_ids_set, name_mapping, primary_key='app_id', rank_column='total_amt'):
    if not app_rank_path or not os.path.exists(app_rank_path):
        return None

    if app_rank_path.lower().endswith('.txt'):
        df = pd.read_csv(app_rank_path, header=0, encoding='utf-8-sig', sep='\x01',
                         quoting=csv.QUOTE_NONE)
    else:
        df = pd.read_csv(app_rank_path, header=0)

    app_rank = df.set_index(primary_key)[rank_column].to_dict()
    name2id_cnt = {}
    for app_id, cnt in app_rank.items():
        if app_id not in app_ids_set:
            continue
        app_name = name_mapping.get(app_id, '')
        if not app_name:
            continue
        if app_name not in name2id_cnt or cnt > name2id_cnt[app_name][1]:
            name2id_cnt[app_name] = (app_id, cnt)
    return name2id_cnt


def normalize_embeddings(embeddings):
    embeddings_float32 = embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings_float32, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized_embeddings = embeddings_float32 / norms
    return normalized_embeddings


def compute_similarity_with_faiss(config, normalized_embeddings, app_ids_str, name_mapping, app_rank=None):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '16'
    faiss.omp_set_num_threads(16)

    # 让 Faiss 自己控制多线程数（默认会用满物理核）
    top_k = config.get('top_k', 200)
    fin_sim_games = config.get('top_n', 200)
    filter_source = config.get('filter_source', False)
    filter_target = config.get('filter_target', False)
    n = len(app_ids_str)
    dim = normalized_embeddings.shape[1]

    names_array = np.array([name_mapping.get(aid, '') for aid in app_ids_str])
    app_rank_best_id = {}
    if app_rank:
        for name, (best_id, _) in app_rank.items():
            app_rank_best_id[name] = best_id

    print(f"🌐 构建 HNSW 索引 (dim={dim}, n={n})，耗时约 2~4 分钟...")
    index = faiss.IndexHNSWFlat(dim, 64)  # M=48 平衡内存与检索精度
    index.hnsw.efConstruction = 256  # 建图质量（越高越准，耗时略增）
    index.hnsw.efSearch = int(top_k * 1.2)  # 检索时扩展节点数（控制召回率）
    chunk_add = 50000
    for start in tqdm(range(0, n, chunk_add), desc="📦 添加向量块", unit="vec"):
        end = min(start + chunk_add, n)
        index.add(normalized_embeddings[start:end].astype('float32'))
    print("HNSW 索引构建完成")

    result = {}
    chunk_size = config.get('chunk_size', 5000)  # HNSW 支持较大 batch
    n_chunks = (n + chunk_size - 1) // chunk_size

    for chunk_idx in tqdm(range(n_chunks), desc="HNSW批量检索", dynamic_ncols=True):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n)
        query_vecs = normalized_embeddings[start_idx:end_idx].astype('float32')

        # 检索 top_k + 1
        D, I = index.search(query_vecs, top_k + 1)

        for i, global_idx in enumerate(range(start_idx, end_idx)):
            source_name = names_array[global_idx]
            similar_games = []
            has_app_names = set()

            for j in range(len(I[i])):
                target_global_idx = I[i][j]
                sim_score = D[i][j]

                if target_global_idx == global_idx or sim_score < 0:
                    continue

                target_name = names_array[target_global_idx]
                if filter_source and target_name == source_name:
                    continue
                if filter_target and target_name in has_app_names:
                    continue
                has_app_names.add(target_name)

                final_target_id = app_ids_str[target_global_idx]
                if app_rank_best_id and target_name in app_rank_best_id:
                    final_target_id = app_rank_best_id[target_name]

                similar_games.append([
                    final_target_id,
                    target_name,
                    round(float(sim_score), 6)
                ])

                if len(similar_games) >= fin_sim_games:
                    break

            result[app_ids_str[global_idx]] = similar_games

    print(f"✓ Faiss HNSW 计算完成，共 {len(result)} 个物品")
    return result


def compute_similarity(config, normalized_embeddings, app_ids_str, name_mapping, app_rank=None):
    """
    优化核心：
    1. 用 np.argpartition 替代 argsort（只取Top-K，不全排序）
    2. 批量处理chunk内所有item的Top-K，消除内层Python循环
    3. 预构建向量化映射数组，减少字典查找
    """

    chunk_size = config.get('chunk_size', 3000)
    sim_games_limit = config.get('top_k', 200)  # 候选集，需 > final top_n 以应对去重
    fin_sim_games = config.get('top_n', 200)
    n = len(app_ids_str)
    filter_source = config.get('filter_source', False)
    filter_target = config.get('filter_target', False)

    # 🔥 预构建向量化映射表（关键优化！）
    # 将字典查找转为数组索引，速度提升10倍+
    app_ids_array = np.array(app_ids_str)
    names_array = np.array([name_mapping.get(aid, '') for aid in app_ids_str])

    # 如果有app_rank，预构建 name->best_id 映射
    app_rank_best_id = {}
    if app_rank:
        for name, (best_id, _) in app_rank.items():
            app_rank_best_id[name] = best_id

    n_chunks = (n + chunk_size - 1) // chunk_size
    print(f"总共 {n} 个物品，分为 {n_chunks} 个块处理，chunk_size={chunk_size}")

    result = {}

    for chunk_idx in tqdm(range(n_chunks), desc="计算相似度块"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n)
        chunk_len = end_idx - start_idx

        # 矩阵乘法计算相似度 (chunk_len, n)
        chunk_similarities = np.dot(
            normalized_embeddings[start_idx:end_idx],
            normalized_embeddings.T
        )

        # 🔥 关键1: 用 argpartition 批量获取每个item的Top-K候选索引
        # 形状: (chunk_len, sim_games_limit)
        k = min(sim_games_limit + 1, n)  # +1防止把自己排除后不够K个
        partitioned = np.argpartition(chunk_similarities, -k, axis=1)[:, -k:]

        # 对每个item的候选集内部排序（只排K个，很快）
        for i in range(chunk_len):
            global_idx = start_idx + i
            candidate_indices = partitioned[i]
            candidate_scores = chunk_similarities[i, candidate_indices]

            # 小范围排序得到最终顺序
            sorted_local_idx = np.argsort(candidate_scores)[::-1]
            sorted_candidates = candidate_indices[sorted_local_idx]
            sorted_scores = candidate_scores[sorted_local_idx]

            # 🔥 关键2: 向量化后处理（去重+映射+app_rank）
            source_name = names_array[global_idx]
            similar_games = []
            has_app_names = set()

            for j_idx, target_global_idx in enumerate(sorted_candidates):
                sim_score = sorted_scores[j_idx]
                if sim_score < 0 or target_global_idx == global_idx:
                    continue

                target_name = names_array[target_global_idx]
                if filter_source and target_name == source_name:
                    continue
                if filter_target and target_name in has_app_names:
                    continue
                has_app_names.add(target_name)

                # 应用app_rank逻辑
                final_target_id = app_ids_str[target_global_idx]
                if app_rank_best_id and target_name in app_rank_best_id:
                    final_target_id = app_rank_best_id[target_name]

                similar_games.append([
                    final_target_id,
                    target_name,
                    round(float(sim_score), 6)
                ])

                if len(similar_games) >= fin_sim_games:
                    break

            result[app_ids_str[global_idx]] = similar_games

        del chunk_similarities  # 及时释放内存

    print(f"✓ 相似度计算完成，共 {len(result)} 个物品")
    return result


def save_results(result, output_csv, name_mapping, top_n=10):
    rows = []
    for item_id, similar_items in tqdm(result.items(), desc="生成CSV"):
        item_name = name_mapping.get(item_id, '')

        row = {
            'item_name': item_name,
            'item_id': item_id
        }

        for rank in range(1, top_n + 1):
            if rank <= len(similar_items):
                target_item_id = similar_items[rank - 1][0]
                target_name = similar_items[rank - 1][1]
                sim_score = similar_items[rank - 1][2]
                row[f'top-{rank}'] = f"{target_name}_{target_item_id}_{sim_score:.6f}"
            else:
                row[f'top-{rank}'] = ''

        rows.append(row)

    df_top = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_top.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"CSV已保存到: {output_csv}")

    return df_top


def save_hive_csv(result, output_hive_csv, top_n=50, chunk_size=50000, has_name_mapping=True):
    if not result:
        print("结果为空，跳过保存。")
        return

    item_ids, sim_items_id, sim_items_score = [], [], []
    chunk_idx = 0
    base_path, ext = os.path.splitext(output_hive_csv)
    out_dir = os.path.dirname(output_hive_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for key, values in tqdm(result.items(), desc="生成Hive CSV"):
        sim_ids, sim_scores = [], []
        for sim_item in values[:top_n]:
            sim_ids.append(sim_item[0])
            sim_scores.append(str(sim_item[-1]))

        item_ids.append(key)
        sim_items_id.append(",".join(sim_ids))
        sim_items_score.append(",".join(sim_scores))

        if len(item_ids) >= chunk_size:
            df = pd.DataFrame({
                'item_id': item_ids,
                'sim_items_id': sim_items_id,
                'sim_items_score': sim_items_score
            })
            chunk_path = f"{base_path}_{chunk_idx}{ext}"
            df.to_csv(chunk_path, encoding='utf-8-sig', sep='|', index=False, header=False)
            print(f"✅ 分片 {chunk_idx} 已保存: {chunk_path} (共 {len(df)} 行)")

            item_ids, sim_items_id, sim_items_score = [], [], []
            chunk_idx += 1

    if item_ids:
        df = pd.DataFrame({
            'item_id': item_ids,
            'sim_items_id': sim_items_id,
            'sim_items_score': sim_items_score
        })
        chunk_path = f"{base_path}_{chunk_idx}{ext}"
        df.to_csv(chunk_path, encoding='utf-8-sig', sep='|', index=False, header=False)
        print(f"分片 {chunk_idx} 已保存: {chunk_path} (共 {len(df)} 行)")
        chunk_idx += 1

    print(f"🎉 全部完成！共生成 {chunk_idx} 个 Hive CSV 分片文件。")


def visualize_top_similarity(df_top, n_rows=10):
    print("\n" + "=" * 80)
    print(f"前{n_rows}行相似结果可视化:")
    print("=" * 80)
    display_cols = ['item_name', 'item_id'] + [f'top-{i}' for i in range(1, 11)]
    available_cols = [col for col in display_cols if col in df_top.columns]
    print(df_top[available_cols].head(n_rows).to_string())


def main():
    parser = argparse.ArgumentParser(description='Embedding相似度计算任务')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()

    config = load_config(args.config)

    npz_path = config['input']['embedding_file']
    csv_path = config['input'].get('id_name_mapping_file', '')
    output_csv = config['output']['output_csv']
    output_hive_csv = config['output'].get('output_hive_csv', '')
    hive_top_n = config['output'].get('hive_top_n', 50)

    primary_key = config['input'].get('primary_key', 'app_id')
    name_column = config['input'].get('name_column', 'app_cn_name')
    npz_emb_key = config['input'].get('npz_emb_key', 'embedding')
    rank_column = config['input'].get('rank_column', 'total_amt')

    print("=" * 60)
    print("Embedding相似度计算任务")
    print("=" * 60)
    print(f"Embedding文件: {npz_path}")
    print(f"ID-名称映射文件: {csv_path}")
    print("=" * 60)

    app_ids, embeddings = load_embedding(npz_path, primary_key, npz_emb_key)
    print(f"共加载 {len(app_ids)} 个embedding")
    print(f"Embedding维度: {embeddings.shape[1]}")

    name_mapping = {}
    if csv_path and os.path.exists(csv_path):
        name_mapping = load_name_mapping(csv_path, primary_key, name_column)
        print(f"共加载 {len(name_mapping)} 个名称映射")
    else:
        print("id_name_mapping_file为空或不存在，跳过名称映射加载")

    app_rank = None
    if name_mapping:
        app_rank_path = config['input'].get('app_rank_file', '')
        app_ids_set = set([str(x) for x in app_ids])
        app_rank = load_app_rank(app_rank_path, app_ids_set, name_mapping, primary_key, rank_column)

        if app_rank is not None:
            print(f"使用app_rank进行筛选，共 {len(app_rank)} 个名称")
        else:
            print("未配置app_rank，跳过筛选逻辑")
    else:
        print("无有效name_mapping，跳过app_rank模块")

    normalized_embeddings = normalize_embeddings(embeddings)
    print("Embedding归一化完成")

    app_ids_str = [str(x) for x in app_ids]

    sim_method = config.get('sim_method', 'base')

    if sim_method == 'faiss':
        result = compute_similarity_with_faiss(config, normalized_embeddings, app_ids_str, name_mapping, app_rank)
    else:
        result = compute_similarity(config, normalized_embeddings, app_ids_str, name_mapping, app_rank)

    df_top = save_results(result, output_csv, name_mapping,
                          top_n=config.get('output', {}).get('top_n', 10))

    if output_hive_csv:
        save_hive_csv(result, output_hive_csv, top_n=hive_top_n, has_name_mapping=bool(name_mapping))

    visualize_top_similarity(df_top, n_rows=config.get('output', {}).get('visualize_rows', 10))

    print("\n任务完成!")


if __name__ == '__main__':
    main()
