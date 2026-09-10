from typing import Dict, Tuple
import logging

import torch
import torch.nn as nn

from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.embedding_modules import EmbeddingModule
from modeling.generic.sequential.loss_modules import LossModule
from modeling.generic.sequential.negative_sampler import NegativesSampler
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const, FeatConst


@ModelRegistry.register(
    req_subs={"EmbeddingModule", "LossModule"},
    opt_subs={"NegativesSampler"})
class MLP_model(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        # 初始化基础组件
        self.embedding_module: EmbeddingModule = self.init_sub_model("EmbeddingModule")
        self.loss_module: LossModule = self.init_sub_model("LossModule")
        self.negative_sampler: NegativesSampler = None if "NegativesSampler" not in model_cfg[Const.SUB_MODELS] \
            else self.init_sub_model("NegativesSampler")
        if self.negative_sampler is not None:
            self.negative_sampler.load_embedding_module(self.embedding_module)
        # 简单的MLP配置
        model_conf = common_hp["model_conf"]
        feat_conf = common_hp["feature_conf"]
        # 获取特征配置
        self.hist_items_key = feat_conf.get("history_items_key", FeatConst.DFLT_HIST_ITEM_KEY)
        self.cand_items_key = feat_conf.get("candidate_items_key", FeatConst.DFLT_CAND_ITEM_KEY)
        self.cand_ratings_key = feat_conf.get("candidate_ratings_column", FeatConst.DFLT_CAND_RATINGS_KEY)
        # MLP超参数
        self.config_embedding_dim = model_conf.get("embedding_dim", 64)  # 保存配置值用于参考
        hidden_dims = model_conf.get("mlp_hidden_dims", [256, 128, 64])
        dropout_rate = model_conf.get("dropout_rate", 0.1)

        # 注意：实际的embedding_dim会在forward中确定
        self.actual_embedding_dim = None  # 将在第一次forward时设置
        self.mlp = None  # 延迟初始化MLP

    def _build_mlp(self, actual_embedding_dim):
        """根据实际embedding维度构建MLP"""
        self.actual_embedding_dim = actual_embedding_dim

        # 计算输入维度：用户历史表示 + 候选商品表示
        input_dim = actual_embedding_dim * 2

        # 构建MLP层
        layers = []
        prev_dim = input_dim
        hidden_dims = [256, 128, 64]  # 可以从配置中读取
        dropout_rate = 0.1  # 可以从配置中读取
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        if next(self.parameters()).device.type != 'cpu':
            device = next(self.parameters()).device
            self.mlp = self.mlp.to(device)

        # 初始化参数
        self.reset_params()

    def reset_params(self):
        """初始化MLP参数"""
        for name, param in self.mlp.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

    def forward(self, model_input: dict, is_train: bool = True) -> torch.Tensor | dict:
        """
        简单MLP前向传播

        Args:
            model_input: 包含历史和候选商品信息的字典
            is_train: 是否为训练模式
        """
        # 获取embedding
        past_embeddings = self.embedding_module.get_ui_embeddings(
            group_name=FeatConst.HIST_PFX, input_features=model_input
        )
        candidate_embeddings = self.embedding_module.get_ui_embeddings(
            group_name=FeatConst.CAND_PFX, input_features=model_input
        )
        user_embeddings = self.embedding_module.get_ui_embeddings(
            group_name=FeatConst.USER_PFX, input_features=model_input
        )

        actual_embedding_dim = past_embeddings.shape[-1]

        if self.mlp is None or self.actual_embedding_dim != actual_embedding_dim:
            self._build_mlp(actual_embedding_dim)

        # 获取候选商品表示
        # candidate_embeddings shape: [batch_size, num_candidates, embedding_dim]
        batch_size, num_candidates, embedding_dim = candidate_embeddings.shape

        # 拼接用户表示和候选商品表示
        mlp_input = torch.cat([user_embeddings, candidate_embeddings],
                              dim=-1)  # [batch_size, num_candidates, embedding_dim*2]

        # reshape为MLP输入格式
        mlp_input = mlp_input.view(-1, embedding_dim * 2)  # [batch_size * num_candidates, embedding_dim*2]

        # MLP前向传播
        scores = self.mlp(mlp_input)  # [batch_size * num_candidates, 1]
        scores = scores.view(batch_size, num_candidates)  # [batch_size, num_candidates]

        # 应用sigmoid激活
        predictions = torch.sigmoid(scores)

        if not is_train or torch.onnx.is_in_onnx_export():
            return {"rerank_score": predictions}
        else:
            predictions_dict = {"rerank_score": predictions}
            # 计算损失
            loss = self.loss_module(
                past_embeddings=None,
                encoded_embeddings=None,
                predictions=predictions_dict,
                model_inputs=model_input,
                negative_sampler=self.negative_sampler
            )
            return loss