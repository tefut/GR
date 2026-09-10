import abc

import torch
from typing import Dict, List, Tuple
from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const


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
class LayerNormEmbeddingPostprocessorLonger(OutputPostprocessorModule):
    """
    L2归一化输出后处理模块，用于对模型输出的嵌入进行L2归一化处理。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._embedding_dim = model_conf.get("item_embedding_dim", 128)
        self._eps: float = model_cfg[Const.HP].get("eps", 1e-7)
        self.layer_norm_input = torch.nn.LayerNorm((self._embedding_dim,), eps=self._eps)

    def forward(
            self,
            output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.layer_norm_input(output_embeddings)
