from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry


@ModelRegistry.register()
class SRNModule(BaseModel):
    """
    Soft Retargeting Network Module - 相似分桶网络模块
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super(SRNModule, self).__init__(model_cfg, common_hp, model_cls_dict)

        # 从超参数中获取配置
        hp = model_cfg.get("hp", {})

        # SRN核心参数
        self.num_bins = hp.get("num_bins", 10)
        self.embedding_size = hp.get("embedding_size", 128)
        self.sim_gate_w = hp.get("sim_gate_w", 10.0)
        self.sim_gate_b = hp.get("sim_gate_b", 9.0)
        self.sim_gate_trainable = hp.get("sim_gate_trainable", True)
        self.gru_units = hp.get("gru_units", None)
        self.binning_type = hp.get("binning_type", "default")

        # 参数校验
        if self.num_bins <= 0:
            raise ValueError(f"The number of similar bins should be positive, current is {self.num_bins}.")
        if self.embedding_size <= 0:
            raise ValueError(f"The embedding size should be positive, current is {self.embedding_size}.")
        if self.gru_units is not None and self.gru_units <= 0:
            raise ValueError(f"The number of gru units should be positive, current is {self.gru_units}.")

        # 计算分桶间隔
        self.interval = 2.0 / self.num_bins

        # 创建相似分桶的embedding表
        self.sim_bins = nn.Embedding(self.num_bins, self.embedding_size)

        # 相似门函数参数
        self.sim_gate_w_param = nn.Parameter(
            torch.tensor([self.sim_gate_w], dtype=torch.float32),
            requires_grad=self.sim_gate_trainable
        )
        self.sim_gate_b_param = nn.Parameter(
            torch.tensor([self.sim_gate_b], dtype=torch.float32),
            requires_grad=self.sim_gate_trainable
        )

        # GRU模块（可选）
        self.gru = None
        if self.gru_units is not None:
            self.gru = nn.GRU(
                input_size=self.embedding_size,
                hidden_size=self.gru_units,
                batch_first=True
            )

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化模型参数"""
        nn.init.uniform_(self.sim_bins.weight, -0.1, 0.1)

    def sim_gate_func(self, cos_sim: torch.Tensor) -> torch.Tensor:
        """
        相似门函数
        """
        w = self.sim_gate_w_param
        b = self.sim_gate_b_param

        numerator = torch.sigmoid(w * cos_sim - b)
        denominator = torch.sigmoid(w - b)

        return numerator / denominator

    def forward(self, inputs: List[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        SRN前向传播 - 修改版本

        Args:
            inputs: [candidate_emb, past_emb] 列表
                   candidate_emb: [batch_size, num_rerank, emb_dim]
                   past_emb: [batch_size, hist_length, emb_dim]
            mask: 序列mask [batch_size, hist_length]

        Returns:
            enhanced_past_embeddings: [batch_size, hist_length, emb_dim]
        """
        if len(inputs) != 2:
            raise ValueError("Inputs should contain exactly 2 tensors: candidate and past embeddings")

        candidate_emb, past_emb = inputs
        batch_size, num_rerank, emb_size = candidate_emb.shape
        batch_size2, hist_length, hist_emb_size = past_emb.shape

        # 验证维度
        if emb_size != self.embedding_size or hist_emb_size != self.embedding_size:
            raise ValueError(
                f"Embedding size mismatch: expected {self.embedding_size}, got {emb_size} and {hist_emb_size}")
        if batch_size != batch_size2:
            raise ValueError(f"Batch size mismatch: {batch_size} vs {batch_size2}")

        # 重新设计思路：对每个历史item，计算其与所有候选item的相似度，
        # 然后基于相似度进行分桶和加权

        # 规范化向量用于余弦相似度计算
        candidate_norm = F.normalize(candidate_emb, dim=-1, eps=1e-3)  # [batch_size, num_rerank, emb_dim]
        past_norm = F.normalize(past_emb, dim=-1, eps=1e-3)  # [batch_size, hist_length, emb_dim]

        # 计算每个历史item与所有候选item的余弦相似度
        cos_sim = torch.matmul(past_norm, candidate_norm.transpose(-1, -2))

        # 对每个历史item，在候选item维度上取最大相似度作为该历史item的代表相似度
        max_cos_sim, _ = torch.max(cos_sim, dim=-1)

        # 分桶类型处理 - Normalized方式（每个历史item独立处理）
        if self.binning_type == 'normalize':
            if mask is not None:
                # 对于有mask的位置，使用一个较大的负值填充
                masked_max_cos_sim = max_cos_sim.masked_fill(~mask, -1e4)
                max_val = masked_max_cos_sim.max(dim=1, keepdim=True)[0]  # [batch_size, 1]
                masked_max_cos_sim = max_cos_sim.masked_fill(~mask, 1e4)
                min_val = masked_max_cos_sim.min(dim=1, keepdim=True)[0]  # [batch_size, 1]
            else:
                max_val = max_cos_sim.max(dim=1, keepdim=True)[0]  # [batch_size, 1]
                min_val = max_cos_sim.min(dim=1, keepdim=True)[0]  # [batch_size, 1]

            # 归一化到[-1, 1]区间
            normalized_cos_sim = 2.0 * (max_cos_sim - min_val) / (max_val - min_val + 1e-6) - 1.0
            binning_cos_sim = normalized_cos_sim
        else:
            # default方式：直接使用原始相似度
            binning_cos_sim = max_cos_sim

        # 映射到分桶索引 [batch_size, hist_length]
        cos_sim_idx = ((binning_cos_sim + 1.0) / self.interval).long()
        cos_sim_idx = torch.clamp(cos_sim_idx, 0, self.num_bins - 1)

        # 获取分桶embedding [batch_size, hist_length, embedding_size]
        bin_embed = self.sim_bins(cos_sim_idx)

        # 应用mask（zero-out）
        if mask is not None:
            bin_embed = bin_embed * mask.unsqueeze(-1).float()  # [batch_size, hist_length, embedding_size]

        # 应用相似门函数
        sim_weight = self.sim_gate_func(max_cos_sim)  # [batch_size, hist_length]
        sim_weight = sim_weight.unsqueeze(-1)  # [batch_size, hist_length, 1]
        weighted_bin_embed = bin_embed * sim_weight  # [batch_size, hist_length, embedding_size]

        # 平均池化作为兴趣向量（这里我们不需要池化，因为我们是逐位置处理的）
        # 直接使用weighted_bin_embed作为增强的embedding
        enhanced_embeddings = weighted_bin_embed

        # 可选GRU兴趣演化建模
        if self.gru is not None:
            gru_output, _ = self.gru(enhanced_embeddings)  # [batch_size, hist_length, gru_units]
            if mask is not None:
                gru_output = gru_output * mask.unsqueeze(-1).float()

            enhanced_embeddings = gru_output

        return enhanced_embeddings