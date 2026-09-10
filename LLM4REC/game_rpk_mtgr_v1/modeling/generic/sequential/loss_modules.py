from __future__ import annotations
import abc
import logging
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from modeling.generic.sequential.negative_sampler import NegativesSampler
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.features import SequentialFeatures
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const


@ModelRegistry.register(req_subs={"LossAggregator"},
                        multi_sel_multi_subs=[{"BinaryCrossEntropyLossForRerankScore",
                                               "CrossEntropyLossForNextActionPred",
                                               "SampledSoftmaxLossForNextItemPred"}])
class LossModule(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        # sub_models必须包含至少一个子Loss模块和一个LossAggregator，且LossAggregator是最后一个sub_models
        # 因此，按照以下顺序加载这些子模块
        sub_models = list(model_cfg[Const.SUB_MODELS].keys())
        self._loss_modules: nn.ModuleList[BaseModel] = nn.ModuleList([
            self.init_sub_model(sub_key)
            for sub_key in sub_models[:-1]
        ])
        self._loss_aggregator = self.init_sub_model(sub_models[-1])

    def forward(
            self,
            past_embeddings: torch.Tensor,
            encoded_embeddings: torch.Tensor,
            predictions: Dict[str, torch.Tensor],
            model_inputs: SequentialFeatures,
            negative_sampler: NegativesSampler
    ) -> Tuple[str, torch.Tensor]:
        losses = []
        for loss_module in self._loss_modules:
            _, loss = loss_module(
                past_embeddings=past_embeddings,
                encoded_embeddings=encoded_embeddings,
                predictions=predictions,
                model_inputs=model_inputs,
                negative_sampler=negative_sampler
            )
            losses.append(loss)
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
class BinaryCrossEntropyLossForRerankScore(BaseModel):
    """
    计算商品打分与标签之间的BCE损失
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.name = "bce_loss_rerank_score"
        model_conf = common_hp.get("model_conf", {})
        train_conf = common_hp.get("train_conf", {})
        self._bf16_mode = (train_conf.get("use_amp", False) and
                           model_conf.get("amp_dtype", "fp16") == "bf16")
        # 判断配置里是否有子模块
        if Const.SUB_MODELS in model_cfg:
            self._loss_weight_modules = nn.ModuleList([
                self.init_sub_model(sub_key)
                for sub_key in model_cfg[Const.SUB_MODELS].keys()
            ])
        else:
            self._loss_weight_modules = None
        self.allowed_prediction_names = "rerank_score"
        self._phase = common_hp.get("train_conf").get("phase", "pretrain")
        self.label_name = common_hp.get("feature_conf").get("candidate_ratings_column")

    def forward(self,
                past_embeddings,
                encoded_embeddings,
                predictions,
                model_inputs,
                negative_sampler) -> Tuple[str, torch.Tensor]:
        if self._phase == "pretrain":
            y_pred, y_aux = predictions
            scores = predictions.get(self.allowed_prediction_names, None)

            if isinstance(self._loss_weight_modules, nn.ModuleList):
                init_mask = None
                for loss_weight in self._loss_weight_modules:
                    _, mask = loss_weight(model_inputs)
                    if init_mask is None:
                        init_mask = mask
                    else:
                        init_mask = init_mask * mask
            else:
                # 没有loss_weight,生成全1的mask，即不做过滤
                init_mask = torch.ones_like(scores)

            labels = model_inputs.get(self.label_name)
            # BF16模式下使用bfloat16避免与BF16预测值的dtype promotion开销
            # BCEWithLogitsLoss内部会自动upcast到FP32进行数值稳定的log-sum-exp计算
            _label_dtype = torch.bfloat16 if self._bf16_mode else torch.float32
            main_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y_pred[init_mask == 1].reshape(-1), (labels[init_mask == 1]).reshape(-1).to(_label_dtype)
            )

            auxiliary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y_aux[init_mask == 1].reshape(-1), (labels[init_mask == 1]).reshape(-1).to(_label_dtype)
            )
            return self.name, main_loss + auxiliary_loss
        else:
            scores = predictions.get(self.allowed_prediction_names, None)
            if scores is None:
                logging.error("No predictions named %s", self.allowed_prediction_names)
            if isinstance(self._loss_weight_modules, nn.ModuleList):
                init_mask = None
                for loss_weight in self._loss_weight_modules:
                    _, mask = loss_weight(model_inputs)
                    if init_mask is None:
                        init_mask = mask
                    else:
                        init_mask = init_mask * mask
            else:
                # 没有loss_weight,生成全1的mask，即不做过滤
                init_mask = torch.ones_like(scores)

            labels = model_inputs.get(self.label_name)
            # BF16模式下使用bfloat16避免与BF16预测值的dtype promotion开销
            _label_dtype = torch.bfloat16 if self._bf16_mode else torch.float32
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                scores[init_mask == 1].reshape(-1),
                (labels[init_mask == 1]).reshape(-1).to(_label_dtype))
            return self.name, loss


@ModelRegistry.register(opt_subs={"DefaultLossMask", "FeatureBasedLossMask", "AGFeatureBasedLossMask"})
class CrossEntropyLossForNextActionPred(LossModule):
    """
    计算action预测和实际action之间的多分类CE损失
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.name = "ce_loss_token"
        # 判断配置里是否有子模块
        if Const.SUB_MODELS in model_cfg:
            self._loss_weight_modules = nn.ModuleList([
                self.init_sub_model(sub_key)
                for sub_key in model_cfg[Const.SUB_MODELS].keys()
            ])
        else:
            self._loss_weight_modules = None
        self.allowed_prediction_names = "next_action_prob"

    def forward(self,
                past_embeddings,
                encoded_embeddings,
                predictions,
                model_inputs,
                negative_sampler) -> Tuple[str, torch.Tensor]:
        scores = predictions.get(self.allowed_prediction_names, None)
        if scores is None:
            logging.error("No predictions named %s", self.allowed_prediction_names)
        if isinstance(self._loss_weight_modules, nn.ModuleList):
            init_mask = None
            for loss_weight in self._loss_weight_modules:
                _, mask = loss_weight(model_inputs)
                if init_mask is None:
                    init_mask = mask
                else:
                    init_mask = init_mask * mask
        else:
            # 没有loss_weight,生成全1的mask，即不做过滤
            init_mask = torch.ones_like(scores)
        labels = model_inputs["ratings"]
        loss = torch.nn.functional.cross_entropy(scores[init_mask == 1], (labels[init_mask == 1]).int())
        return self.name, loss


@ModelRegistry.register(opt_subs={"DefaultLossMask", "FeatureBasedLossMask", "AGFeatureBasedLossMask"})
class SampledSoftmaxLossForNextItemPred(LossModule):
    """
    计算下一个商品预测的infoNCE损失
    {
    "hp": {"softmax_temperature": float}
    }
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.name = "sampledsoftmax_loss_item"
        self._softmax_temperature: float = model_cfg[Const.HP].get("softmax_temperature", 0.5)
        model_conf = common_hp.get("model_conf", {})
        train_conf = common_hp.get("train_conf", {})
        self._bf16_mode = (train_conf.get("use_amp", False) and
                           model_conf.get("amp_dtype", "fp16") == "bf16")
        # 判断配置里是否有子模块
        if Const.SUB_MODELS in model_cfg:
            self._loss_weight_modules = nn.ModuleList([
                self.init_sub_model(sub_key)
                for sub_key in model_cfg[Const.SUB_MODELS].keys()
            ])
        else:
            self._loss_weight_modules = None
        self.allowed_prediction_names = "rerank_score"

    def dot_product(
            self,
            input_embeddings: torch.Tensor,
            item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_embeddings: (B, D,) or (B * r, D) x float.
            item_embeddings: (1, X, D) or (B, X, D) x float.

        Returns:
            (B, X) x float (or (B * r, X) x float).
        """

        if item_embeddings.size(0) == 1:
            return torch.mm(input_embeddings, item_embeddings.squeeze(0).t()), {}  # [B, X]
        elif input_embeddings.size(0) != item_embeddings.size(0):
            B, X, D = item_embeddings.size()
            return torch.bmm(input_embeddings.view(B, -1, D), item_embeddings.permute(0, 2, 1)).view(-1, X)
        else:
            return torch.bmm(item_embeddings, input_embeddings.unsqueeze(2)).squeeze(2)

    def jagged_forward(
            self,
            output_embeddings: torch.Tensor,
            supervision_embeddings: torch.Tensor,
            supervision_weights: torch.Tensor,
            negatives_sampler: NegativesSampler,
            model_inputs: SequentialFeatures,
            positive_ids: torch.Tensor
    ) -> Tuple[str, torch.Tensor]:

        positive_ids = model_inputs['past_ids']
        # sampledsoftmax计算所需的处理
        positive_ids = positive_ids[:, 1:].long().reshape(-1)
        sampled_ids, sampled_negative_embeddings = negatives_sampler(
            model_inputs=model_inputs,
            positive_ids=positive_ids
        )

        if hasattr(negatives_sampler, 'module'):
            positive_embeddings = negatives_sampler.module.normalize_embeddings(supervision_embeddings)
        else:
            positive_embeddings = negatives_sampler.normalize_embeddings(supervision_embeddings)

        positive_embeddings = positive_embeddings.unsqueeze(1)

        # replace 
        positive_logits = self.dot_product(
            input_embeddings=output_embeddings,  # [B, D] = [N', D]
            item_embeddings=positive_embeddings,  # [N', D] -> [N', 1, D]
        ) / self._softmax_temperature  # [0]
        sampled_negatives_logits = self.dot_product(
            input_embeddings=output_embeddings,  # [N', D]
            item_embeddings=sampled_negative_embeddings,  # [N', R, D]
        )  # [N', R]  # [0]

        jagged_loss = -F.log_softmax(
            torch.cat([positive_logits, sampled_negatives_logits], dim=1), dim=1
        )[:, 0]
        eps = 1e-3
        return self.name, (jagged_loss * supervision_weights).sum() / (supervision_weights.sum() + eps)

    def forward(
            self,
            past_embeddings,
            encoded_embeddings,
            predictions,
            model_inputs,
            negative_sampler,
    ) -> Tuple[str, torch.Tensor]:
        scores = predictions.get(self.allowed_prediction_names, None)
        if scores is None:
            logging.error("No predictions named %s", self.allowed_prediction_names)
        if isinstance(self._loss_weight_modules, list):
            init_mask = None
            for loss_weight in self._loss_weight_modules:
                mask = loss_weight(model_inputs)
                if init_mask is None:
                    init_mask = mask
                else:
                    init_mask = init_mask * mask
        elif isinstance(self._loss_weight_modules, BaseModel):
            init_mask = self._loss_weight_modules(model_inputs)
        else:
            # 没有loss_weight,即为None
            init_mask = torch.ones_like(scores)
        # sampledsoftmax计算所需的处理
        B, N = model_inputs['past_ids'].size()
        encoded_embeddings = encoded_embeddings[:, :-1, :]
        past_embeddings = past_embeddings[:, 1:, :]
        loss_weights = init_mask.reshape(B, N)[:, 1:].to(torch.bfloat16 if self._bf16_mode else torch.float32)

        D = encoded_embeddings.size(-1)
        if encoded_embeddings.size() != past_embeddings.size():
            raise ValueError(f"编码嵌入尺寸 {encoded_embeddings.size()} 与过去嵌入尺寸 {past_embeddings.size()} 不匹配")

        return self.jagged_forward(
            output_embeddings=encoded_embeddings.reshape(-1, D),
            supervision_embeddings=past_embeddings.reshape(-1, D),
            supervision_weights=loss_weights.reshape(-1),
            negatives_sampler=negative_sampler,
            model_inputs=model_inputs
        )


@ModelRegistry.register()
class SumLossAggregator(LossAggregator):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

    def forward(self, losses: List[torch.Tensor]):
        total_loss = 0.
        for loss_value in losses:
            total_loss += loss_value
        return total_loss
