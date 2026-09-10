#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQKMeans Constrained 模型核心函数
=================================

提供平衡约束的残差K-Means量化实现。
使用 k-means-constrained 库确保每层聚类大小均匀分布。
"""

import logging
import time

import numpy as np

try:
    from k_means_constrained import KMeansConstrained

    HAS_CONSTRAINED = True
except ImportError:
    HAS_CONSTRAINED = False
    KMeansConstrained = None


def balanced_kmeans_level_constrained(
        X, K, max_iter=100, tol=1e-7, random_state=None, verbose=False
):
    """
    使用 KMeansConstrained 实现平衡K-Means聚类

    Args:
        X: 输入数据 (N, d)
        K: 聚类数量
        max_iter: 最大迭代次数
        tol: 收敛容差
        random_state: 随机种子
        verbose: 是否打印详细信息

    Returns:
        labels: (N,) 聚类标签
        centroids: (K, d) 聚类中心
    """
    start_time = time.time()
    n, d = X.shape
    X = np.array(X, dtype=np.float32, copy=True, order='C')

    # 计算最小和最大簇大小
    min_size = max(1, n // K - 1)
    max_size = n // K + 1

    if verbose:
        logging.info(
            "    Starting constrained K-means with K=%d, n=%d, d=%d", K, n, d
        )
        logging.info(
            "    Cluster size constraints: [%d, %d]", min_size, max_size
        )

    # 使用 k-means-constrained
    kmeans = KMeansConstrained(
        n_clusters=K,
        size_min=min_size,
        size_max=max_size,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_init=3,
        verbose=verbose,
        n_jobs=8,
    )

    # 训练并获取标签
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    logging.info(
        "[Time] balanced_kmeans_level_constrained (K=%d): %.2fs",
        K, time.time() - start_time
    )

    if verbose:
        unique, counts = np.unique(labels, return_counts=True)
        logging.info(
            "    Cluster sizes: min=%d, max=%d, mean=%.1f",
            counts.min(), counts.max(), counts.mean()
        )

    return labels, centroids


def residual_kmeans_constrained(
        X, K, L, max_iter=300, tol=1e-4, random_state=None, verbose=False
):
    """
    残差K-Means（使用平衡聚类约束）

    Args:
        X: 输入数据 (N, d)
        K: 每层聚类数量（int 或 list）
        L: 层级数量
        max_iter: K-Means 最大迭代次数
        tol: 收敛容差
        random_state: 随机种子
        verbose: 打印详细信息

    Returns:
        codes_all: (L, N) 每层整数编码
        codebooks: L 个 codebook 列表，每个 (K, d)
        recon: 重构数据 (N, d)
    """
    total_start = time.time()
    n, d = X.shape
    Ks = ([K] * L) if isinstance(K, int) else list(K)

    X = np.ascontiguousarray(X.astype(np.float32))
    R = X.copy()  # copy 默认是可写的
    codes_all = np.empty((L, n), dtype=np.int32)
    codebooks = []

    for l in range(L):
        level_start = time.time()
        k_l = Ks[l]
        if verbose:
            mse_before = np.mean(R ** 2)
            logging.info(
                "\n=== Level %d/%d | K=%d ===", l + 1, L, k_l
            )
            logging.info("  Residual MSE before clustering: %.6f", mse_before)

        # 为子层级生成随机种子
        seed_l = None
        if random_state is not None:
            seed_l = int(
                np.random.RandomState(random_state + l).randint(0, 2 ** 31 - 1)
            )

        codes_l, C_l = balanced_kmeans_level_constrained(
            R, k_l, max_iter=max_iter, tol=tol, random_state=seed_l, verbose=verbose
        )

        codes_all[l] = codes_l
        codebooks.append(C_l)

        # 从残差中减去重构部分
        R -= C_l[codes_l]

        logging.info("[Time] Level %d: %.2fs", l + 1, time.time() - level_start)
        if verbose:
            mse_after = np.mean(R ** 2)
            logging.info("  Residual MSE after Level %d: %.6f", l + 1, mse_after)

    recon = X - R
    logging.info(
        "[Time] residual_kmeans_constrained total: %.2fs",
        time.time() - total_start
    )

    if verbose:
        total_mse = np.mean((X - recon) ** 2)
        logging.info("\nFinal reconstruction MSE: %.6f", total_mse)

    return codes_all, codebooks, recon


def check_constrained_library():
    """
    检查 k-means-constrained 库是否可用

    Returns:
        bool: 库是否可用
    """
    return HAS_CONSTRAINED
