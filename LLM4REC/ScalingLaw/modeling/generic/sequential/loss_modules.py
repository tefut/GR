from __future__ import annotations
import abc
import logging
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from modeling.generic.sequential.negative_sampler import NegativesSampler
from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const


@ModelRegistry.register(req_subs={"LossAggregator"},
                        multi_sel_multi_subs=[{"BinaryCrossEntropyLossForRerankScoreLonger",
                                               "EncodedEmbeddingsL2Loss", "PastEmbeddingsL2Loss"}])
class LossModule(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        # sub_models必须包含至少一个子Loss模块和一个LossAggregator，且LossAggregator是最后一个sub_models
        # 因此，按照以下顺序加载这些子模块
        sub_models = list(model_cfg[Const.SUB_MODELS].keys())
        self._loss_modules: List[BaseModel] = nn.ModuleList([
            self.init_sub_model(sub_key)
            for sub_key in sub_models[:-1]
        ])
        self._loss_aggregator = self.init_sub_model(sub_models[-1])

    def forward(
            self,
            past_embeddings: torch.Tensor,
            encoded_embeddings: torch.Tensor,
            predictions: Dict[str, torch.Tensor],
            model_inputs: Dict,
            negative_sampler: NegativesSampler
    ) -> Tuple[str, torch.Tensor]:
        losses = dict()
        for loss_module in self._loss_modules:
            loss_name, loss = loss_module(past_embeddings=past_embeddings,
                                          encoded_embeddings=encoded_embeddings,
                                          predictions=predictions,
                                          model_inputs=model_inputs,
                                          negative_sampler=negative_sampler)
            losses[loss_name] = loss
        aggregated_loss = self._loss_aggregator(losses)
        return aggregated_loss


class LossAggregator(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

    @abc.abstractmethod
    def forward(
            self,
            losses: torch.Tensor
    ) -> torch.Tensor:
        pass


@ModelRegistry.register(opt_subs={"DefaultLossMask", "FeatureBasedLossMask", "AGFeatureBasedLossMask"})
class BinaryCrossEntropyLossForRerankScoreLonger(BaseModel):
    """
    计算商品打分与标签之间的BCE损失
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.name = "bce_loss_rerank_score"
        # 判断配置里是否有子模块
        if Const.SUB_MODELS in model_cfg:
            self._loss_weight_modules = nn.ModuleList([
                self.init_sub_model(sub_key)
                for sub_key in model_cfg[Const.SUB_MODELS].keys()
            ])
        else:
            self._loss_weight_modules = None
        self.allowed_prediction_names = "rerank_score"

    def forward(self,
                past_embeddings,
                encoded_embeddings,
                predictions,
                model_inputs,
                negative_sampler) -> Tuple[str, torch.Tensor]:
        scores = predictions.get(self.allowed_prediction_names, None)
        # 对score做截断，防止过大或过小值导致loss出问题
        eps = 1e-7
        scores = torch.nan_to_num(scores)
        scores = torch.clamp(scores, eps, 1 - eps)
        if scores is None:
            logging.error("No predictions named %s", self.allowed_prediction_names)
        labels = model_inputs["label"]
        loss = torch.nn.functional.binary_cross_entropy(scores.reshape(-1), labels.reshape(-1).float())
        return self.name, loss


@ModelRegistry.register()
class BinaryCrossEntropyLossForRerankBaseline(BaseModel):
    """
    计算商品打分与标签之间的BCE损失
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.name = "bce_loss_rerank_score"
        self.allowed_prediction_names = "rerank_score"

    def forward(self,
                predictions,
                labels,
                ) -> Tuple[str, torch.Tensor]:
        scores = predictions
        # 对score做截断，防止过大或过小值导致loss出问题
        eps = 1e-7
        scores = torch.nan_to_num(scores)
        scores = torch.clamp(scores, eps, 1 - eps)
        loss = torch.nn.functional.binary_cross_entropy(scores.reshape(-1), labels.reshape(-1).float())
        return loss


@ModelRegistry.register()
class PastEmbeddingsL2Loss(BaseModel):
    """
    计算从EmbeddingModule出来的序列embedding的L2正则
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.name = "input_embedding_l2_loss"

    def forward(self,
                past_embeddings,
                encoded_embeddings,
                predictions,
                model_inputs,
                negative_sampler) -> Tuple[str, torch.Tensor]:
        device = past_embeddings.device
        zero_embedding = torch.zeros_like(past_embeddings, device=device, requires_grad=False)
        loss = torch.nn.functional.mse_loss(past_embeddings, zero_embedding)
        return self.name, loss


@ModelRegistry.register()
class EncodedEmbeddingsL2Loss(BaseModel):
    """
    计算从Transformer出来的序列embedding的L2正则
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.name = "encoded_embedding_l2_loss"

    def forward(self,
                past_embeddings,
                encoded_embeddings,
                predictions,
                model_inputs,
                negative_sampler) -> Tuple[str, torch.Tensor]:
        device = encoded_embeddings.device
        zero_embedding = torch.zeros_like(encoded_embeddings, device=device, requires_grad=False)
        loss = torch.nn.functional.mse_loss(encoded_embeddings, zero_embedding)
        return self.name, loss


@ModelRegistry.register()
class SumLossAggregator(LossAggregator):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

    def forward(self, losses: Dict[str, torch.Tensor]):
        total_loss = 0.
        for _, loss_value in losses.items():
            total_loss += loss_value
        return total_loss


@ModelRegistry.register()
class WeightedSumLossAggregator(LossAggregator):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        WeightedSumLossAggregatorConfig = model_cfg[Const.HP]
        self.loss_weights = dict()
        for k, v in WeightedSumLossAggregatorConfig.items():
            self.loss_weights[k] = v

    def forward(self, losses: Dict[str, torch.Tensor]):
        total_loss = 0.
        for loss_name, loss_value in losses.items():
            loss_weight = self.loss_weights.get(loss_name, 1.)
            total_loss += loss_value * loss_weight
        return total_loss
