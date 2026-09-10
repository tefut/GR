#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS ResidualQuantizer 训练和推理工具
=========================================

提供 FAISS 残差量化器的训练、编码、保存和加载功能。
设计用于支持 rqkmeans 和 rqkmeans_plus 模型类型。

主要函数：
- train_faiss_rq(): 训练 FAISS 残差量化器
- encode_with_rq(): 使用量化器编码数据
- sinkhorn_uniform_mapping(): Sinkhorn 均匀映射
- save_codebook_npz(): 导出 codebook 为 .npz
- save_faiss_index(): 导出 .faiss 索引
- load_pretrained_codebook(): 加载 .npz 到 RQVAE 模型
"""

import json
import os

import faiss
import numpy as np
import torch
from tqdm import tqdm


# ========================================
# 辅助函数
# ========================================

def pairwise_sq_dists_batch(X, C, C_norm2=None):
    """
    计算批量欧氏距离平方

    Parameters:
        X: (B, d) 查询向量
        C: (K, d) 中心点

    Returns:
        (B, K) 距离矩阵
    """
    if C_norm2 is None:
        C_norm2 = np.sum(C * C, axis=1)  # (K,)
    X_norm2 = np.sum(X * X, axis=1, keepdims=True)  # (B, 1)
    dots = X @ C.T  # (B, K)
    return X_norm2 + C_norm2[None, :] - 2.0 * dots


def get_first_nbits(rq):
    """获取量化器的 nbits 值"""
    if isinstance(rq.nbits, int):
        return rq.nbits
    return int(faiss.vector_to_array(rq.nbits).ravel()[0])


# ========================================
# 核心函数
# ========================================

def train_faiss_rq(data, num_levels=3, codebook_size=256, verbose=True, rq=None):
    """
    训练 FAISS ResidualQuantizer

    Parameters:
        data: (N, d) numpy array，训练数据
        num_levels: 量化层数
        codebook_size: 每层 codebook 大小（必须是 2 的幂）
        verbose: 是否打印训练信息
        rq: 可选的预创建 faiss.ResidualQuantizer 对象，如果为 None 则创建新的

    Returns:
        faiss.ResidualQuantizer 对象
    """
    import time
    N, d = data.shape
    if verbose:
        print("Training FAISS ResidualQuantizer")
        print(f"  data={N}  dim={d}  levels={num_levels}  "
              f"codebook={codebook_size}  total_codes={codebook_size ** num_levels:,}")

    nbits = int(np.log2(codebook_size))
    if rq is None:
        rq = faiss.ResidualQuantizer(d, num_levels, nbits)
        rq.train_type = faiss.ResidualQuantizer.Train_default
        rq.max_beam_size = 1

    if verbose:
        print("  Start training...")
        start_time = time.time()

    rq.train(np.ascontiguousarray(data.astype(np.float32)))

    if verbose:
        elapsed = time.time() - start_time
        print(f"  training completed in {elapsed:.2f}s\n")
    return rq


def unpack_rq_codes(codes, nbits, num_levels):
    """
    解包 FAISS 的位压缩 codes 为整数索引数组

    Parameters:
        codes: (N, M_bytes) uint8 array，来自 rq.compute_codes
        nbits: 每层比特数（如 512 -> 9 bits）
        num_levels: 量化层数

    Returns:
        (N, num_levels) int32 array，解包后的索引
    """
    N = codes.shape[0]
    # FAISS 使用 Little Endian 打包
    packed_ints = np.zeros(N, dtype=np.int64)
    for i in range(codes.shape[1]):
        packed_ints |= codes[:, i].astype(np.int64) << (8 * i)
    unpacked_codes = np.zeros((N, num_levels), dtype=np.int32)
    mask = (1 << nbits) - 1  # e.g., 9 bits 的 mask 是 511 (0x1FF)
    for i in range(num_levels):
        unpacked_codes[:, i] = (packed_ints >> (i * nbits)) & mask
    return unpacked_codes


def encode_with_rq(rq, data, codebook_size, verbose=True):
    """
    使用 FAISS 量化器编码数据

    Parameters:
        rq: faiss.ResidualQuantizer 对象
        data: (N, d) numpy array，待编码向量
        codebook_size: 每层 codebook 大小
        verbose: 是否打印编码信息

    Returns:
        (N, num_levels) int32 array，编码后的索引
    """
    data = np.ascontiguousarray(data.astype(np.float32))
    nbits = int(np.log2(codebook_size))
    if verbose:
        print(f"Encoding {data.shape[0]} vectors ...")
    codes_packed = rq.compute_codes(data)
    if nbits % 8 == 0:
        codes = codes_packed.astype(np.int32)
    else:
        codes = unpack_rq_codes(codes_packed, nbits, rq.M)
    if codes_packed.ndim == 1:
        n_bytes = (rq.M * nbits + 7) // 8
        codes_packed = codes_packed.reshape(-1, n_bytes)
    codes = codes.astype(np.int32)
    if verbose:
        print(f"  done, codes.shape={codes.shape}\n")
    return codes


def get_rq_codebooks(rq):
    """
    获取量化器的 codebooks

    Parameters:
        rq: faiss.ResidualQuantizer 对象

    Returns:
        (M, K, d) numpy array，codebooks
    """
    M, d = rq.M, rq.d
    nbits0 = get_first_nbits(rq)
    K = 1 << nbits0
    cb_flat = faiss.vector_to_array(rq.codebooks).astype(np.float32)
    return cb_flat.reshape(M, K, d)


def compute_residuals_upto_level(rq, data, codes, upto_level, codebooks=None):
    """
    计算到指定层级的残差

    Parameters:
        rq: faiss.ResidualQuantizer 对象
        data: (N, d) 原始数据
        codes: (N, M) 编码结果
        upto_level: 计算到第几层（不包含）
        codebooks: 可选的 codebooks数组

    Returns:
        (N, d) 残差向量
    """
    if codebooks is None:
        codebooks = get_rq_codebooks(rq)
    residuals = np.ascontiguousarray(data.astype(np.float32)).copy()
    for l in range(upto_level):
        residuals -= codebooks[l][codes[:, l]]
    return residuals


def estimate_tau(residuals, centroids, sample_size=4000,
                 percentile=90, min_tau=1e-6):
    """
    估计 Sinkhorn 正则化参数 tau

    Parameters:
        residuals: (N, d) 残差向量
        centroids: (K, d) 中心点
        sample_size: 采样大小
        percentile: 百分位数
        min_tau: 最小 tau 值

    Returns:
        float, 估计的 tau 值
    """
    N = residuals.shape[0]
    idx = np.random.choice(N, size=min(sample_size, N), replace=False)
    X = residuals[idx]
    Cn2 = np.sum(centroids * centroids, axis=1)
    D = pairwise_sq_dists_batch(X, centroids, Cn2)
    spread = np.percentile(D - D.min(axis=1, keepdims=True),
                           percentile, axis=1)
    tau = float(np.median(spread) * 0.1)
    return max(tau, min_tau)


def sinkhorn_balance_level(residuals, centroids, capacities=None, *,
                           batch_size=8192, iters=30, tau=None,
                           verbose=True, topk=32, seed=42):
    """
    使用 Sinkhorn 算法平衡单层量化器

    Parameters:
        residuals: (N, d) 残差向量
        centroids: (K, d) 中心点
        capacities: 可选的容量数组
        batch_size: 批大小
        iters: Sinkhorn 迭代次数
        tau: 正则化参数
        verbose: 是否打印信息
        topk: topk 采样
        seed: 随机种子

    Returns:
        (N,) int32 array，重新分配的中心索引
    """
    import ot
    rng = np.random.RandomState(seed)
    N, d = residuals.shape
    K = centroids.shape[0]

    if capacities is None:
        capacities = np.full(K, N // K, dtype=np.int64)
        capacities[: (N % K)] += 1
    capacities = capacities.astype(np.int64)

    if tau is None:
        tau = estimate_tau(residuals, centroids)
    if verbose:
        print(f"  Sinkhorn level: N={N}  K={K}  tau={tau:.5g}  "
              f"iters={iters}  batch={batch_size}")

    a = np.ones(N) / N
    b = capacities / float(N)
    Cn2 = np.sum(centroids * centroids, axis=1)

    def cost_fun(X):
        return pairwise_sq_dists_batch(X, centroids, Cn2)

    D_full = cost_fun(residuals).astype(np.float64)
    P = ot.sinkhorn(a, b, D_full, tau)

    remaining = capacities.copy()
    assign = np.empty(N, dtype=np.int32)
    order = np.arange(N)
    rng.shuffle(order)
    for i in order:
        probs = P[i]
        if topk and topk < K:
            cand = np.argpartition(-probs, topk - 1)[:topk]
            cand = cand[np.argsort(-probs[cand])]
        else:
            cand = np.argsort(-probs)
        chosen = -1
        for c in cand:
            if remaining[c] > 0:
                chosen = c
                break
        if chosen < 0:
            c = int(np.argmax(probs))
            if remaining[c] == 0:
                c = int(np.argmin(remaining))
            chosen = c
        remaining[chosen] -= 1
        assign[i] = chosen

    if verbose:
        used = capacities - remaining
        print(f"    level balanced: min={used.min()}  max={used.max()}")

    return assign


def sinkhorn_uniform_mapping(rq, data, codes, *, batch_size=8192,
                             iters=30, tau=None, verbose=True,
                             topk=32, seed=42):
    """
    使用 Sinkhorn 均匀映射平衡所有层级

    Parameters:
        rq: faiss.ResidualQuantizer 对象
        data: (N, d) 原始数据
        codes: (N, M) 原始编码
        batch_size: 批大小
        iters: Sinkhorn 迭代次数
        tau: 正则化参数
        verbose: 是否打印信息
        topk: topk 采样
        seed: 随机种子

    Returns:
        (N, M) int32 array，平衡后的编码
    """
    codebooks = get_rq_codebooks(rq)
    N, M = codes.shape
    K = codebooks.shape[1]

    codes_bal = codes.copy()
    for l in range(M):
        if verbose:
            print(f"\n=== Sinkhorn uniform mapping  level {l + 1}/{M} ===")
        residuals = compute_residuals_upto_level(
            rq, data, codes_bal, upto_level=l, codebooks=codebooks)

        capacities = np.full(K, N // K, dtype=np.int64)
        capacities[: (N % K)] += 1

        new_ids = sinkhorn_balance_level(
            residuals, codebooks[l], capacities=capacities,
            batch_size=batch_size, iters=iters, tau=tau,
            verbose=verbose, topk=topk, seed=seed + l)

        codes_bal[:, l] = new_ids
    return codes_bal


def analyze_codes(codes, title="", verbose=True):
    """
    分析编码分布

    Parameters:
        codes: (N, M) 编码数组
        title: 标题
        verbose: 是否打印信息
    """
    N, M = codes.shape
    if verbose:
        if title:
            print(title)
        print(f"  total={N}")
        for l in range(M):
            print(f"  L{l + 1}: unique={len(np.unique(codes[:, l]))}")
        combos = len(set(map(tuple, codes)))
        print(f"  unique full-paths={combos}  "
              f"collision_rate={1 - combos / N:.4f}")
    return


def save_indices_json(codes, path, use_prefix=True):
    """
    保存索引到 JSON 文件

    Parameters:
        codes: (N, M) 编码数组
        path: 输出路径
        use_prefix: 是否使用前缀格式（如 <a_0>）
    """
    tpl = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    idx = {}
    for i, code in enumerate(codes):
        if use_prefix:
            idx[i] = [tpl[j].format(int(c)) for j, c in enumerate(code)]
        else:
            idx[i] = [int(c) for c in code]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(idx, f, indent=2)
    print("Saved indices:", path)


# ========================================
# 保存函数
# ========================================

def save_codebook_npz(rq, path, verbose=True):
    """
    保存量化器的 codebooks 为 .npz 文件

    Parameters:
        rq: faiss.ResidualQuantizer 对象
        path: 输出路径
        verbose: 是否打印信息
    """
    codebooks = get_rq_codebooks(rq)
    M = codebooks.shape[0]

    save_dict = {}
    for l in range(M):
        save_dict[f"codebook_{l}"] = codebooks[l]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **save_dict)

    if verbose:
        print(f"Saved codebooks to {path}")
        print(f"  levels={M}, codebook_size={codebooks.shape[1]}, dim={codebooks.shape[2]}")


def save_faiss_index(rq, ckpt_dir, verbose=True):
    """
    保存量化器为 .faiss 索引文件

    Parameters:
        rq: faiss.ResidualQuantizer 对象
        ckpt_dir: checkpoint 目录
        verbose: 是否打印信息
    """
    try:
        nbits_val = get_first_nbits(rq)
        index = faiss.IndexResidualQuantizer(rq.d, rq.M, nbits_val)
        index.rq = rq
        index.is_trained = True
        path = os.path.join(ckpt_dir, "rq.index.faiss")
        faiss.write_index(index, path)
        if verbose:
            print(f"Saved faiss index to {path}")
    except Exception as e:
        if verbose:
            print(f"Save faiss index failed: {e}")


# ========================================
# 加载函数
# ========================================

def load_pretrained_codebook(npz_path, model, device):
    """
    从 .npz 文件加载预训练的 FAISS codebook 到 RQVAE 模型

    Parameters:
        npz_path: str, .npz 文件路径，包含 codebook_0, codebook_1, ... 数组
        model: RQVAE 模型实例，codebook 将被加载到 model.rq.vq_layers
        device: torch device，模型所在的设备

    Raises:
        FileNotFoundError: npz 文件不存在
        KeyError: npz 中缺少必需的 codebook 键
        ValueError: codebook 维度不匹配
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Pretrained codebook file not found: {npz_path}")

    data = np.load(npz_path)

    # 检查必需的键
    num_levels = len(model.rq.vq_layers)
    for l in range(num_levels):
        key = f"codebook_{l}"
        if key not in data:
            raise KeyError(
                f"Missing required key '{key}' in npz file. "
                f"Expected keys: " + ", ".join(f"codebook_{i}" for i in range(num_levels))
            )

    # 加载每个层的 codebook
    for l in range(num_levels):
        key = f"codebook_{l}"
        cb = data[key]  # shape: (K, d)

        # 验证维度
        vq_layer = model.rq.vq_layers[l]
        expected_K, expected_d = vq_layer.embedding.weight.shape

        if cb.shape[0] != expected_K or cb.shape[1] != expected_d:
            raise ValueError(
                f"Codebook dimension mismatch at level {l}: "
                f"got shape {cb.shape}, expected ({expected_K}, {expected_d})"
            )

        # 加载到模型
        vq_layer.embedding.weight.data = torch.from_numpy(cb).to(device)

    # 确保 codebook 与模型在同一设备
    model.to(device)

    print(f"Loaded pretrained codebook from {npz_path} ({num_levels} levels)")
    return model


def load_faiss_quantizer(npz_path):
    """
    加载预训练的 FAISS 量化器（用于推理）

    Parameters:
        npz_path: .npz 文件路径

    Returns:
        faiss.ResidualQuantizer 对象
    """
    data = np.load(npz_path)

    # 从 codebook_0 的形状推断参数
    codebook_0 = data["codebook_0"]
    K, d = codebook_0.shape
    nbits = int(np.log2(K))

    # 推断层数
    num_levels = 0
    while f"codebook_{num_levels}" in data:
        num_levels += 1

    # 创建量化器
    rq = faiss.ResidualQuantizer(d, num_levels, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1

    # 设置 codebooks
    codebooks = np.zeros((num_levels, K, d), dtype=np.float32)
    for l in range(num_levels):
        codebooks[l] = data[f"codebook_{l}"]

    rq.codebooks = faiss.cast_bytes_to_vector(faiss.vector_to_array(codebooks))

    return rq


# ========================================
# 主函数（用于独立运行）
# ========================================

def main():
    """独立运行 FAISS-RQ 训练和评估"""
    import argparse
    parser = argparse.ArgumentParser(
        description="FAISS-RQ + Sinkhorn uniform mapping")
    parser.add_argument("--dataset", default="Industrial_and_Scientific")
    parser.add_argument("--data_path", type=str, default=None)

    parser.add_argument("--num_levels", type=int, default=3)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--uniform", action="store_true",
                        help="enable Sinkhorn uniform mapping")
    parser.add_argument("--iters", type=int, default=30,
                        help="Sinkhorn iterations")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--output_root", default="../data")
    args = parser.parse_args()

    if args.data_path is not None:
        data_path = args.data_path
    else:
        data_path = f"../data/Amazon/index/{args.dataset}.emb-qwen-td.npy"

    out_dir = os.path.join(args.output_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{args.dataset}.faiss-rq.index.json")
    out_faiss = out_json.replace(".json", ".faiss")

    print("loading:", data_path)
    data = np.load(data_path)
    print("shape:", data.shape)

    rq = train_faiss_rq(data, args.num_levels, args.codebook_size)
    codes_raw = encode_with_rq(rq, data, args.codebook_size, verbose=True)

    analyze_codes(codes_raw, "Before balancing:")

    if args.uniform:
        codes_bal = sinkhorn_uniform_mapping(
            rq, data, codes_raw,
            batch_size=args.batch_size,
            iters=args.iters,
            verbose=True)
        analyze_codes(codes_bal, "After  balancing:")
        codes_final = codes_bal
    else:
        codes_final = codes_raw

    save_indices_json(codes_final, out_json, use_prefix=True)

    try:
        nbits_val = get_first_nbits(rq)
        index = faiss.IndexResidualQuantizer(rq.d, rq.M, nbits_val)
        index.rq = rq
        index.is_trained = True
        faiss.write_index(index, out_faiss)
        print("Saved faiss quantizer:", out_faiss)
    except Exception as e:
        print("save faiss index failed:", e)


if __name__ == "__main__":
    main()
