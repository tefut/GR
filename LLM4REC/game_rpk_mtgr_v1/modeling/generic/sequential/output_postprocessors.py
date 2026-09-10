import abc

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const, FeatConst


class OutputPostprocessorModule(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
            self,
            output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        pass


@ModelRegistry.register()
class L2NormEmbeddingPostprocessor(OutputPostprocessorModule):
    """
    L2归一化输出后处理模块，用于对模型输出的嵌入进行L2归一化处理。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._embedding_dim: int = model_conf.get("item_embedding_dim", FeatConst.ITEM_EMB_DIM)
        if model_conf.get('use_user_embeddings_for_rerank', False):
            self._embedding_dim *= 2
        self._eps: float = model_cfg[Const.HP].get("eps", Const.EPS)

    def forward(
            self,
            output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        output_embeddings = output_embeddings[..., :self._embedding_dim]
        # L2 norm在FP16下squared_sum极易溢出，需FP32计算
        # BF16有与FP32相同的指数范围，squared_sum不会溢出，无需升FP32
        orig_dtype = output_embeddings.dtype
        if output_embeddings.dtype == torch.float16:
            output_embeddings = output_embeddings.float()
        squared_sum = torch.sum(output_embeddings ** 2, dim=-1, keepdim=True)
        result = torch.div(output_embeddings, torch.clamp(torch.sqrt(torch.clamp(squared_sum, 0.0) + self._eps),
                                                        min=self._eps))
        return result.to(orig_dtype)


@ModelRegistry.register()
class LayerNormEmbeddingPostprocessor(OutputPostprocessorModule):
    """
    层归一化输出后处理模块，用于对模型输出的嵌入进行层归一化处理。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._embedding_dim: int = model_conf.get("item_embedding_dim", FeatConst.ITEM_EMB_DIM)
        self._eps: float = model_cfg[Const.HP].get("eps", Const.EPS)
        self.layer_norm_input = nn.LayerNorm((self._embedding_dim,), eps=self._eps)

    def forward(
            self,
            output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        output_embeddings = output_embeddings[..., :self._embedding_dim]
        return self.layer_norm_input(output_embeddings)