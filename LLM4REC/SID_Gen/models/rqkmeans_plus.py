#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQKMeans+ 残差编码器包装器
===========================

提供 ResidualEncoderWrapper 包装器和 apply_rqkmeans_plus_strategy() 函数。
用于 rqkmeans_plus 模型类型的残差连接和零初始化策略。

残差连接公式：Z = X + MLP(X)
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn


class ResidualEncoderWrapper(nn.Module):
    """
    残差编码器包装器

    将原始 encoder 包装为残差连接形式：
        output = input + mlp(input)

    这允许模型学习 identity mapping，同时保持量化能力。
    用于 RQKMeans+ 模式。
    """

    def __init__(self, original_encoder):
        """
        Parameters:
            original_encoder: 原始的 encoder 模块
        """
        super().__init__()
        self.mlp = original_encoder

    def forward(self, x):
        """前向传播：残差连接"""
        return x + self.mlp(x)


def apply_rqkmeans_plus_strategy(model, codebook_path, device):
    """
    应用 RQKMeans+ 策略到 RQVAE 模型

    策略包括：
    1. 将 encoder 包装为残差连接形式 (Z = X + MLP(X))
    2. 对 encoder 最后一层进行零初始化
    3. 从 .npz 文件加载预训练的 codebooks

    Parameters:
        model: RQVAE 模型实例
        codebook_path: 预训练 codebook .npz 文件路径
        device: torch device

    Returns:
        包装后的模型

    Raises:
        FileNotFoundError: codebook 文件不存在
    """
    logging.info(">>> [RQ-Kmeans+] Strategy: Applying Residual Connection & Warm-start...")

    # 1. 包装 encoder 为残差连接形式
    if hasattr(model, 'encoder'):
        model.encoder = ResidualEncoderWrapper(model.encoder)
        model.encoder.to(device)
        logging.info("    [Structure] Encoder wrapped with Residual Connection (Z = X + MLP(X))")
    else:
        logging.error("    [Error] Could not find 'encoder' in model.")
        return model

    # 2. 对 encoder 最后一层进行零初始化
    logging.info("    [Init] Applying Zero-Initialization to Encoder's last layer...")

    last_linear = None
    raw_mlp = model.encoder.mlp

    if hasattr(raw_mlp, 'mlp_layers'):
        modules = list(raw_mlp.mlp_layers.modules())
    else:
        modules = list(raw_mlp.modules())

    for m in reversed(modules):
        if isinstance(m, nn.Linear):
            last_linear = m
            break

    if last_linear:
        with torch.no_grad():
            last_linear.weight.fill_(0.0)
            if last_linear.bias is not None:
                last_linear.bias.fill_(0.0)
        logging.info(f"    [Init] Zero-init applied to Linear layer: {last_linear}")
    else:
        logging.warning("    [Warning] Could not find last Linear layer to zero-init.")

    # 3. 加载预训练 codebooks
    if not os.path.exists(codebook_path):
        raise FileNotFoundError(f"{codebook_path} not found")

    logging.info(f"    [Weights] Loading codebooks from {codebook_path}")
    npz_data = np.load(codebook_path)

    target_layers = None
    if hasattr(model, 'rq') and hasattr(model.rq, 'vq_layers'):
        target_layers = model.rq.vq_layers

    if target_layers:
        success_count = 0
        for i, layer in enumerate(target_layers):
            emb_layer = layer.embedding if hasattr(layer, 'embedding') else layer

            key = f'codebook_{i}'
            if key in npz_data:
                centroids = npz_data[key]
                with torch.no_grad():
                    emb_layer.weight.data.copy_(torch.from_numpy(centroids).to(device))
                success_count += 1
                logging.info(f"      -> Loaded Codebook Level {i}")

        if success_count == 0:
            logging.warning("      -> No codebooks loaded! Check .npz keys.")
    else:
        logging.error("    [Error] Could not locate VQ layers.")

    return model
