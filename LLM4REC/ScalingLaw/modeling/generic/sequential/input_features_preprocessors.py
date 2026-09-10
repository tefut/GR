import abc
import math
from typing import Dict, Tuple, List
import logging
import torch
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.initialization import truncated_normal
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
from utils.common_utils import weird_division


class InputFeaturesPreprocessorModule(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

    @abc.abstractmethod
    def forward(
            self,
            past_ids: List[torch.Tensor],
            num_rerank: int,
            user_feature_embs: torch.Tensor,
            item_feature_embs: torch.Tensor,
            seq_feature_embs: torch.Tensor,
    ) -> torch.Tensor:
        pass


@ModelRegistry.register()
class UserItemRatingInputFeaturePreprocessorLonger(InputFeaturesPreprocessorModule):
    """
    用户-物品-评分输入特征预处理模块, 用于处理用户、物品和评分的特征。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        feat_conf = common_hp["feature_conf"]
        seq_feature_conf = feat_conf.get("seq_feature_columns")

        seq_lens = [
            max(subcfg["length"] for subcfg in feature_dict.values())
            for feature_dict in seq_feature_conf.values()
        ]
        max_seq_len = max(seq_lens)
        self.seq_lens = seq_lens
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)
        self._pos_emb = torch.nn.Embedding(sum(seq_lens) + 3, self._embedding_dim)
        self._seq_type_emb = torch.nn.Embedding(len(seq_lens) + 1, self._embedding_dim)
        self._dropout_rate: float = model_cfg[Const.HP].get("embedding_dropout_rate", 0.0)
        self._emb_dropout = torch.nn.Dropout(p=self._dropout_rate)

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb, mean=0.0, std=math.sqrt(weird_division(1.0, self._embedding_dim)),
        )

    def forward(
            self,
            past_ids: List[torch.Tensor],
            num_rerank: int,
            user_feature_embs: torch.Tensor,
            item_feature_embs: torch.Tensor,
            seq_feature_embs: torch.Tensor,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # -------- 获取额外参数 --------
        model_inputs = kwargs.get("model_inputs", None)
        past_lengths = kwargs.get("past_lengths", None)
        deep_outputs = kwargs.get("deep_outputs", None)
        """
        前向传播方法, 用于处理用户、物品和评分的特征。

        :param past_ids: 历史序列的张量, 列表里面每个项是一个商品id序列，形状为(B, S_i), 其中S_i是每个子序列i的长度,。
        :param user_feature_embs: 用户特征embedding张量, 形状为(B, D), D是用户特征embedding的维度。
        :param item_feature_embs: 歌曲特征embedding张量, 形状为(B, N, D), D是用户特征embedding的维度。
        :param seq_feature_embs: 序列特征embedding张量, , 列表里面每个项是一个序列，形状为(B, S, D), D是用户特征embedding的维度。
        :return: past_lengths: 形状为(B, K), K为子序列的数量, 向量里每个元素表示该子序列有效长度。
                 user_embeddings: 新的序列, 长度历史序列长度总和 + 1 + 2 * num_rerank。
        """
        B, N, D = item_feature_embs.size()
        device = past_lengths.device

        # 形状(1, S)
        pos_ids = torch.cat([torch.arange(n, device=device) for n in self.seq_lens]).unsqueeze(0)
        type_ids = torch.cat([torch.ones(n, device=device, dtype=torch.int) *
                              i for i, n in enumerate(self.seq_lens)]).unsqueeze(0)
        # 形状(B, S, D)
        pos_embs = self._pos_emb(pos_ids).repeat(B, 1, 1)
        type_embs = self._seq_type_emb(type_ids).repeat(B, 1, 1)
        seq_feature_embs = seq_feature_embs + pos_embs + type_embs

        deep_pos_ids = torch.ones((1, num_rerank), dtype=torch.int, device=device)
        deep_pos_embs = self._pos_emb(deep_pos_ids).repeat(B, 1, 1)
        deep_outputs = deep_outputs + deep_pos_embs

        x_offsets = None
        seq_offsets = None

        #         # 对seq_emb进行序列重排 seq_1, pad1, seq_2, pad2, seq_3, pad3... -> seq_1, seq_2, seq_3, pad1, pad2, pad3...
        past_sum = torch.sum(past_lengths, dim=1)
        x_offsets = past_sum + num_rerank
        x_offsets = torch.cumsum(x_offsets, dim=0)
        x_offsets = torch.cat((torch.tensor([0], device=device), x_offsets))

        B, L, D = seq_feature_embs.shape
        device = seq_feature_embs.device

        ids = torch.cat(past_ids, dim=-1)  # [B, 900]
        B, L = ids.shape
        D = seq_feature_embs.size(-1)

        # 1. 有效位置 mask：非 0 为 1，0 为 0
        valid = (ids != 0).to(torch.long)  # [B, L]

        # 2. 每个位置的原始下标 j = 0..L-1
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, -1)  # [B, L]

        key = (1 - valid) * L + pos  # [B, L]

        # 4. 对 key 做排序，拿到每个 batch 的重排下标 perm
        _, perm = torch.sort(key, dim=1)  # [B, L]

        # 5. 用 perm 对 seq_feature_embs 在维度 1 上做 gather
        perm_expanded = perm.unsqueeze(-1).expand(B, L, D)  # [B, L, D]
        seq_feature_embs = torch.gather(seq_feature_embs, 1, perm_expanded)  # [B, L, D]

        # 拼接用户和历史序列token：deep_1, deep_2, ,..., (hist_i1,hist_i2,hist_i3,...), (hist_i'1,hist_i'2,hist_i'3,...),...
        # 形状 (B, N + S, D)
        whole_seq_embeddings = torch.cat(
            [deep_outputs,
             seq_feature_embs
             ], dim=1)
        whole_seq_embeddings = self._emb_dropout(whole_seq_embeddings)

        return whole_seq_embeddings, x_offsets, seq_offsets

    def prefill_forward(
            self,
            past_ids: List[torch.Tensor],
            num_rerank: int,
            user_feature_embs: torch.Tensor,
            item_feature_embs: torch.Tensor,
            seq_feature_embs: torch.Tensor,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        past_lengths = kwargs.get("past_lengths", None)

        B, N, D = item_feature_embs.size()
        device = past_lengths.device

        # 形状(1, S)
        pos_ids = torch.cat([torch.arange(n, device=device) for n in self.seq_lens]).unsqueeze(0)
        type_ids = torch.cat([torch.ones(n, device=device, dtype=torch.int) *
                              i for i, n in enumerate(self.seq_lens)]).unsqueeze(0)
        # 形状(B, S, D)
        pos_embs = self._pos_emb(pos_ids).repeat(B, 1, 1)
        type_embs = self._seq_type_emb(type_ids).repeat(B, 1, 1)
        seq_feature_embs = seq_feature_embs + pos_embs + type_embs

        x_offsets = None
        seq_offsets = None

        # 对seq_emb进行序列重排 seq_1, pad1, seq_2, pad2, seq_3, pad3... -> seq_1, seq_2, seq_3, pad1, pad2, pad3...
        past_sum = torch.sum(past_lengths, dim=1)
        x_offsets = past_sum + num_rerank
        x_offsets = torch.cumsum(x_offsets, dim=0)
        x_offsets = torch.cat((torch.tensor([0], device=device), x_offsets))

        B, L, D = seq_feature_embs.shape
        device = seq_feature_embs.device

        ids = torch.cat(past_ids, dim=-1)  # [B, 900]
        B, L = ids.shape
        D = seq_feature_embs.size(-1)
        valid = (ids != 0).to(torch.long)  # [B, L]
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, -1)  # [B, L]

        key = (1 - valid) * L + pos  # [B, L]
        _, perm = torch.sort(key, dim=1)  # [B, L]
        perm_expanded = perm.unsqueeze(-1).expand(B, L, D)  # [B, L, D]
        seq_feature_embs = torch.gather(seq_feature_embs, 1, perm_expanded)  # [B, L, D]

        return seq_feature_embs, x_offsets, seq_offsets

    def decode_forward(
            self,
            past_ids: List[torch.Tensor],
            num_rerank: int,
            user_feature_embs: torch.Tensor,
            item_feature_embs: torch.Tensor,
            seq_feature_embs: torch.Tensor,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        past_lengths = kwargs.get("past_lengths", None)
        deep_outputs = kwargs.get("deep_outputs", None)
        B, N, D = item_feature_embs.size()
        device = past_lengths.device

        deep_pos_ids = torch.ones((1, num_rerank), dtype=torch.int, device=device)
        deep_pos_embs = self._pos_emb(deep_pos_ids).repeat(B, 1, 1)
        deep_outputs = deep_outputs + deep_pos_embs

        x_offsets = None
        seq_offsets = None

        return deep_outputs, x_offsets, seq_offsets
