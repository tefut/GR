import logging
from typing import Dict

import torch
import torch.nn as nn

from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.utils.constants import Const
from modeling.model_registry import ModelRegistry


@ModelRegistry.register(multi_sel_multi_subs=[{"FeedForwardModuleForRerankScore",
                                               "FeedForwardModuleForNextActionPred"}])
class FeedForwardModule(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        if Const.SUB_MODELS not in model_cfg:
            logging.error("You must assign at least one sub module for FeedForwardModule in the config.")

        self.prediction_modules = nn.ModuleList([
            self.init_sub_model(sub_key)
            for sub_key in model_cfg[Const.SUB_MODELS].keys()
        ])

    def forward(self, x: torch.Tensor, flag: bool = None, return_logits: bool = False):
        predictions = dict()
        for ffn in self.prediction_modules:
            if flag is not None:
                pred_name, pred = ffn(x, flag, return_logits=return_logits)
            else:
                pred_name, pred = ffn(x, return_logits=return_logits)
            predictions[pred_name] = pred
        return predictions


@ModelRegistry.register()
class FeedForwardModuleForRerankScore(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.name = "rerank_score"
        input_dim = model_cfg['hp']['input_dim']
        self.negative_sample_ratio = model_cfg[Const.HP].get("negative_sample_ratio", 0.005)
        # 预计算 log(ratio) 作为 buffer，校正时等价于 logits 偏移 log(ratio)
        self.register_buffer("log_ratio",
                             torch.tensor(float(self.negative_sample_ratio)).log_(),
                             persistent=False)

        _eps: float = model_cfg[Const.HP].get("eps", Const.EPS)

        self.feed_forward = torch.nn.Linear(in_features=input_dim, out_features=input_dim)
        self.out_layer = torch.nn.Linear(in_features=input_dim, out_features=1)
        self.layer_norm_1 = nn.LayerNorm([input_dim], eps=_eps)
        self.layer_norm_2 = nn.LayerNorm([input_dim], eps=_eps)
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor, flag: bool = None, return_logits: bool = False):
        x = x + self.act(self.feed_forward(self.layer_norm_1(x)))
        x = self.out_layer(self.layer_norm_2(x))
        logits = x.squeeze(-1)
        if return_logits:
            return self.name, logits
        if flag:
            # 对数域等价：sigmoid(z)/(sigmoid(z)+(1-sigmoid(z))/r) = sigmoid(z+ln(r))
            # 整体升fp32计算并保持fp32返回，避免小概率值(如1e-7)在fp16下溢出为0
            x = torch.sigmoid(logits.float() + self.log_ratio)
        else:
            x = torch.sigmoid(logits)
        return self.name, x


@ModelRegistry.register()
class FeedForwardModuleForNextActionPred(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self.name = "next_action_prob"
        input_dim = model_cfg['hp']['input_dim']
        output_dim = model_conf.get("num_ratings", 5)
        self.negative_sample_ratio = model_cfg[Const.HP].get("negative_sample_ratio", 0.005)
        self.register_buffer("log_ratio",
                             torch.tensor(float(self.negative_sample_ratio)).log_(),
                             persistent=False)

        _eps: float = model_cfg[Const.HP].get("eps", Const.EPS)

        self.feed_forward = torch.nn.Linear(in_features=input_dim, out_features=input_dim)
        self.out_layer = torch.nn.Linear(in_features=input_dim, out_features=output_dim + 1)
        self.layer_norm_1 = nn.LayerNorm([input_dim], eps=_eps)
        self.layer_norm_2 = nn.LayerNorm([input_dim], eps=_eps)
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor, flag: bool = None, return_logits: bool = False):
        x = x + self.act(self.feed_forward(self.layer_norm_1(x)))
        x = self.out_layer(self.layer_norm_2(x))
        if return_logits:
            return self.name, x
        if flag:
            # 对数域等价：sigmoid(z)/(sigmoid(z)+(1-sigmoid(z))/r) = sigmoid(z+ln(r))
            # 整体升fp32计算并保持fp32返回，避免小概率值在fp16下溢出为0
            x = torch.sigmoid(x.float() + self.log_ratio)
        else:
            x = torch.sigmoid(x)
        return self.name, x


@ModelRegistry.register()
class LinearModuleForRerankScore(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.name = "rerank_score"
        input_dim = model_cfg['hp']['input_dim']
        self.pred_linear = torch.nn.Linear(in_features=input_dim, out_features=1)
        self.negative_sample_ratio = model_cfg[Const.HP].get("negative_sample_ratio", 0.005)
        self.register_buffer("log_ratio",
                             torch.tensor(float(self.negative_sample_ratio)).log_(),
                             persistent=False)

    def forward(self, x: torch.Tensor, flag: bool = None, return_logits: bool = False):
        x = self.pred_linear(x)
        logits = x.squeeze(-1)
        if return_logits:
            return self.name, logits
        if flag:
            # 对数域等价：sigmoid(z)/(sigmoid(z)+(1-sigmoid(z))/r) = sigmoid(z+ln(r))
            # 整体升fp32计算并保持fp32返回，避免小概率值在fp16下溢出为0
            x = torch.sigmoid(logits.float() + self.log_ratio)
        else:
            x = torch.sigmoid(logits)
        return self.name, x