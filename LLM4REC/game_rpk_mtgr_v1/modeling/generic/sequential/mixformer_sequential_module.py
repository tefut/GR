"""
MixFormerSequentialModule: 将 MixFormerModule 包装为 Standard SequentialModule 接口。

MTGR 的 generate_user_embeddings 调用 self.sequence_model(x=..., x_offsets=...)
但 MixFormerModule 接受 x=dict 而非 flat tensor。本模块存储来自前序
MixFormerFeaturePreprocessor 输出的 emb_dict, 在 forward 中将 flat tensor 替换
为 dict 输入, 并确保返回值形状与 MTGR 下游期望一致。
"""

from typing import Dict, List, Optional, Tuple

import torch

from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.mixformer import MixFormerModule
from modeling.generic.sequential.transformers import TransformerCacheState
from modeling.model_registry import ModelRegistry


@ModelRegistry.register(req_hp=True, req_subs={"MixFormerBlock"})
class MixFormerSequentialModule(BaseModel):
    """
    包装 MixFormerModule 以匹配 MTGR SequentialModule 的 forward 接口。

    负责:
    1. 在 init 时创建 MixFormerModule 实例 (作为 self.mixformer_module)。
    2. 外部 (MixFormerModel) 通过 set_emb_dict() 注入 emb_dict。
    3. forward 时用 emb_dict 替代 flat tensor 传给 MixFormerModule。
    4. 从 dict 结果中提取 candidate 部分作为最终输出。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        self._emb_dict: Optional[Dict[str, torch.Tensor]] = None
        self._num_rerank = common_hp.get("data_loader_conf", {}).get("num_rerank", 400)

        # 创建底层的 MixFormerModule
        # 注意: MixFormerModule 注册时使用的 key 是 "MixFormerModule"
        # 但这里我们作为 SequentialModule 注册, 需要在 sub_models 中配置
        # MixFormerBlock 以通过 init_sub_model 创建
        self.mixformer_module: MixFormerModule = self.init_sub_model("MixFormerModule")

    def set_emb_dict(self, emb_dict: Dict[str, torch.Tensor]) -> None:
        """由 MixFormerModel.process_single_act_seq 注入预处理后的 dict。"""
        self._emb_dict = emb_dict

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: torch.Tensor,
        invalid_attn_mask: Optional[torch.Tensor] = None,
        past_lengths: Optional[torch.Tensor] = None,
        num_rerank: int = 0,
        cache: Optional[List[TransformerCacheState]] = None,
        delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
        return_cache_states: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, TransformerCacheState]:
        """
        包装 forward。

        参数签名匹配 MTGR 的 SequentialModule.forward:
            x: 被忽略, 会使用 self._emb_dict 替代
        返回:
            item_embeddings: torch.Tensor [B, 1+N+M, D] 或 [B, M, D]
            cache_states: TransformerCacheState
        """
        if self._emb_dict is None:
            raise RuntimeError(
                "emb_dict not set. Call set_emb_dict() before forward()."
            )

        result_dict, cache_states = self.mixformer_module(
            x=self._emb_dict,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            attn_mask=invalid_attn_mask,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            cache=cache,
            delta_x_offsets=delta_x_offsets,
            return_cache_states=return_cache_states,
        )

        return result_dict, cache_states