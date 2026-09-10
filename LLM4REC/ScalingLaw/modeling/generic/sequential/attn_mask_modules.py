import torch
from typing import Dict

import torch.nn.functional as F
import torch.nn as nn
from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const


@ModelRegistry.register(multi_sel_multi_subs=[{"CausalAttentionMaskLONGER"}])
class AttentionMaskModule(BaseModel):
    """
    "AttentionMaskModule": {
        "type": ["CausalAttentionMask", "TimeAttentionMask"],
        "cfg": {}
    }

    :param model_conf:
    :param model_factory:
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.attention_mask_modules = nn.ModuleList([
            self.init_sub_model(sub_key)
            for sub_key in model_cfg[Const.SUB_MODELS].keys()
        ])

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank):
        init_mask = None
        for attn_module in self.attention_mask_modules:
            mask = attn_module(model_inputs=model_inputs, max_seq_len=max_seq_len, num_rerank=num_rerank)
            if init_mask is None:
                init_mask = mask
            else:
                init_mask = init_mask * mask
        attn_mask = init_mask.detach().clone()
        return attn_mask

    def prefill_forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank):
        for attn_module in self.attention_mask_modules:
            mask_prefill = attn_module(model_inputs=model_inputs, max_seq_len=max_seq_len, num_rerank=num_rerank)
        mask_prefill = mask_prefill.detach().clone()
        return mask_prefill


    def decode_forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank):
        for attn_module in self.attention_mask_modules:
            mask_decode = attn_module(model_inputs=model_inputs, max_seq_len=max_seq_len, num_rerank=num_rerank)
        mask_decode = mask_decode.detach().clone()
        return mask_decode


@ModelRegistry.register()
class CausalAttentionMaskLONGER(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        LONGER causal attention mask
        token placement: user, item, bev_seq_1, bev_seq_2, bev_seq_3 ...
        (reverse chronological order)
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        feat_conf = common_hp["feature_conf"]
        seq_feature_conf = feat_conf.get("seq_feature_columns")

        seq_lens = [
            max(subcfg["length"] for subcfg in feature_dict.values())
            for feature_dict in seq_feature_conf.values()
        ]
        self.seq_lens = seq_lens

        self._infer_items_key = feat_conf.get('infer_items_key', 'item_id')

    def init_idx_mask(self, max_seq_len, num_rerank, device):
        """
        假设有3个候选商品,两个总长度均为5的子序列，其中第一个子序列有效长度为2，第二个子序列有效长度为4.
        idx  0  1  2  3  4  5  6  7  8  9  a    b  c   d   e   f   g
         0 [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         1  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         2  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         3  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         4  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         5  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         6  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         7  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         8  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         9  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         a  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         b  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         c  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         d  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         e  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         f  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         g  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
        """
        max_len = num_rerank + max_seq_len
        indices = torch.arange(max_len, device=device)
        t = indices.expand(max_len, max_len)
        return t

    def init_candidate_mask(self, max_seq_len, num_rerank, device):
        max_len = num_rerank + max_seq_len
        mask = torch.zeros((max_len, max_len), device=device)
        num_rerank_idx = torch.arange(num_rerank, device=device)
        mask[num_rerank_idx, num_rerank_idx] = 1
        return mask

    def fill_seq_mask(self, indices, start, end, num_rerank):
        return (indices >= start) * (indices < end) * (indices.t() < num_rerank)

    def construct_block_mask(self, indices, start, end):
        return (indices >= start) * (indices.t() >= start) * (indices < end) * (indices.t() < end)

    def forward(self, model_inputs: Dict, max_seq_len, num_rerank):
        """
        构造transformer的causal mask。对应序列重排后。
        1. deep_output可以看到自己对应的所有的子序列
        2. userxitem可以看到所有的子序列
        3. 子序列左边新右边旧，序列间互不可见，序列内使用上三角的causal mask
        序列中的token包括deep_output,以及多个子历史序列[seq^i],排列方式如下：
        deep_output_1, deep_output_2,...,[seq^1], [seq^2],..,0,0,0,...
        其中[seq^i]表示第i个子序列的有效长度，填充的0会放到序列尾部。
        假设有2个候选商品,两个总长度均为5的子序列，其中第一个子序列有效长度为2，第二个子序列有效长度为4.
        则第0,1个token为deep_output，2-3个为子序列1的token,第4到7个为子序列b
        idx  0  1  2  3  4  5  6  7  8  9  a  b
         0 [[1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         1  [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         2  [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         3  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         4  [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         5  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
         6  [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         7  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         8  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         9  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         a  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         b  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        """
        # 形状(B, num_seq)
        past_lengths = model_inputs['past_lengths']
        device = model_inputs[self._infer_items_key].device
        B = past_lengths.size(0)
        S = past_lengths.size(1)
        seq_len_sum = sum(max_seq_len)
        L = num_rerank + seq_len_sum
        # 初始化一个上三角矩阵和idx掩码
        t = torch.arange(L, device=device).repeat(L).view(L, L)
        triu = (t.t() <= t).float()
        idx_mask = self.init_idx_mask(seq_len_sum, num_rerank, device)
        candidate_mask = self.init_candidate_mask(seq_len_sum, num_rerank, device)

        # 循环生成
        mask_list = []
        for b in range(B):
            mask = candidate_mask
            past_length = past_lengths[b]
            past_start = num_rerank
            tristart = num_rerank
            len_seq_num = len(max_seq_len)
            for seq_i in range(len_seq_num):
                past_end = past_start + past_length[seq_i]
                mask = mask + self.fill_seq_mask(idx_mask, past_start, past_end, num_rerank)
                past_start = past_start + max_seq_len[seq_i]
                triend = tristart + past_length[seq_i]
                small_tru = triu * self.construct_block_mask(idx_mask, tristart, triend)
                mask += small_tru
                tristart = tristart + max_seq_len[seq_i]
            mask_list.append(mask)
        mask = torch.stack(mask_list)

        return mask.float()


@ModelRegistry.register()
class CausalAttentionMaskLONGERV2(BaseModel):
    """
    V2版本mask为适配有序列重排，构建方式与 CausalAttentionMaskLONGER 仅有细微区别
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        LONGER causal attention mask
        token placement: user, item, bev_seq_1, bev_seq_2, bev_seq_3 ...
        (reverse chronological order)
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        feat_conf = common_hp["feature_conf"]
        seq_feature_conf = feat_conf.get("seq_feature_columns")

        seq_lens = [
            max(subcfg["length"] for subcfg in feature_dict.values())
            for feature_dict in seq_feature_conf.values()
        ]
        self.seq_lens = seq_lens

        self._infer_items_key = feat_conf.get('infer_items_key', 'item_id')

    def init_idx_mask(self, max_seq_len, num_rerank, device):
        """
        假设有3个候选商品,两个总长度均为5的子序列，其中第一个子序列有效长度为2，第二个子序列有效长度为4.
        idx  0  1  2  3  4  5  6  7  8  9  a    b  c   d   e   f   g
         0 [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         1  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         2  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         3  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         4  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         5  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         6  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         7  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         8  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         9  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         a  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         b  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         c  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         d  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         e  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         f  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         g  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
        """
        max_len = num_rerank + max_seq_len
        indices = torch.arange(max_len, device=device)
        t = indices.expand(max_len, max_len)
        return t

    def init_candidate_mask(self, max_seq_len, num_rerank, device):
        max_len = num_rerank + max_seq_len
        mask = torch.zeros((max_len, max_len), device=device)
        num_rerank_idx = torch.arange(num_rerank, device=device)
        mask[num_rerank_idx, num_rerank_idx] = 1
        return mask

    def fill_seq_mask(self, indices, start, end, num_rerank):
        return (indices >= start) * (indices < end) * (indices.t() < num_rerank)

    def construct_block_mask(self, indices, start, end):
        return (indices >= start) * (indices.t() >= start) * (indices < end) * (indices.t() < end)

    def forward(self, model_inputs: Dict, max_seq_len, num_rerank):
        """
        构造transformer的causal mask。对应序列重排后。
        1. deep_output可以看到自己对应的所有的子序列
        2. userxitem可以看到所有的子序列
        3. 子序列左边新右边旧，序列间互不可见，序列内使用上三角的causal mask
        序列中的token包括deep_output,以及多个子历史序列[seq^i],排列方式如下：
        deep_output_1, deep_output_2,...,[seq^1], [seq^2],..,0,0,0,...
        其中[seq^i]表示第i个子序列的有效长度，填充的0会放到序列尾部。
        假设有2个候选商品,两个总长度均为5的子序列，其中第一个子序列有效长度为2，第二个子序列有效长度为4.
        则第0,1个token为deep_output，2-3个为子序列1的token,第4到7个为子序列b
        idx  0  1  2  3  4  5  6  7  8  9  a  b
         0 [[1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         1  [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         2  [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         3  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         4  [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         5  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
         6  [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         7  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         8  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         9  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         a  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         b  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        """
        # 形状(B, num_seq)
        past_lengths = model_inputs['past_lengths']
        device = model_inputs[self._infer_items_key].device
        B = past_lengths.size(0)
        S = past_lengths.size(1)
        seq_len_sum = sum(max_seq_len)
        L = num_rerank + seq_len_sum
        # 初始化一个上三角矩阵和idx掩码
        t = torch.arange(L, device=device).repeat(L).view(L, L)
        triu = (t.t() <= t).float()
        idx_mask = self.init_idx_mask(seq_len_sum, num_rerank, device)
        candidate_mask = self.init_candidate_mask(seq_len_sum, num_rerank, device)

        # 循环生成
        mask_list = []
        for b in range(B):
            mask = candidate_mask
            past_length = past_lengths[b]
            past_start = num_rerank
            tristart = num_rerank
            for seq_i in range(len(max_seq_len)):
                past_end = past_start + past_length[seq_i]
                mask = mask + self.fill_seq_mask(idx_mask, past_start, past_end, num_rerank)
                past_start = past_start + past_length[seq_i]
                triend = tristart + past_length[seq_i]
                small_tru = triu * self.construct_block_mask(idx_mask, tristart, triend)
                mask += small_tru
                tristart = tristart + past_length[seq_i]
            mask_list.append(mask)
        mask = torch.stack(mask_list)

        return mask.float()

    def prefill_forward(self, model_inputs: Dict, max_seq_len, num_rerank):
        past_lengths = model_inputs['past_lengths']
        device = model_inputs[self._infer_items_key].device
        B_prefill = past_lengths.size(0)
        seq_len_sum = sum(max_seq_len)
        L_prefill = num_rerank + seq_len_sum
        t_prefill = torch.arange(L_prefill, device=device).\
            repeat(L_prefill).view(L_prefill, L_prefill)
        triu = (t_prefill.t() <= t_prefill).float()
        idx_mask = self.init_idx_mask(seq_len_sum,
                                      num_rerank,
                                      device)
        candidate_mask = self.init_candidate_mask(seq_len_sum,
                                                  num_rerank,
                                                  device)

        mask_list = []
        for P in range(B_prefill):
            mask_prefill = candidate_mask
            past_length = past_lengths[P]
            past_start = num_rerank
            tristart_prefill = num_rerank
            for seq_i in range(len(max_seq_len)):
                past_end = past_start + past_length[seq_i]
                mask_prefill = mask_prefill + self.fill_seq_mask(idx_mask, past_start,
                                                 past_end, num_rerank)
                past_start = past_start + past_length[seq_i]
                triend_prefill = tristart_prefill + past_length[seq_i]
                small_tru = triu * self.construct_block_mask(idx_mask,
                                                             tristart_prefill, triend_prefill)
                mask_prefill += small_tru
                tristart_prefill = tristart_prefill + past_length[seq_i]
            mask_list.append(mask_prefill)
        mask_prefill = torch.stack(mask_list)
        mask_prefill = mask_prefill[:, -seq_len_sum:, -seq_len_sum:]

        return mask_prefill.float()

    def decode_forward(self, model_inputs: Dict, max_seq_len, num_rerank):
        past_lengths = model_inputs['past_lengths']
        device = model_inputs[self._infer_items_key].device
        B_decode = past_lengths.size(0)
        seq_len_sum = sum(max_seq_len)
        L_decode = num_rerank + seq_len_sum
        t_decode = torch.arange(L_decode, device=device).\
            repeat(L_decode).view(L_decode, L_decode)
        triu = (t_decode.t() <= t_decode).float()
        idx_mask = self.init_idx_mask(seq_len_sum,
                                      num_rerank,
                                      device)
        candidate_mask = self.init_candidate_mask(seq_len_sum,
                                                  num_rerank,
                                                  device)

        mask_list = []
        for D in range(B_decode):
            mask_decode = candidate_mask
            past_length = past_lengths[D]
            past_start = num_rerank
            tristart_decode = num_rerank
            for seq_i in range(len(max_seq_len)):
                past_end = past_start + past_length[seq_i]
                mask_decode = mask_decode + self.fill_seq_mask(idx_mask, past_start,
                                                 past_end, num_rerank)
                past_start = past_start + past_length[seq_i]
                triend_decode = tristart_decode + past_length[seq_i]
                small_tru = triu * self.construct_block_mask(idx_mask,
                                                             tristart_decode, triend_decode)
                mask_decode += small_tru
                tristart_decode = tristart_decode + past_length[seq_i]
            mask_list.append(mask_decode)
        mask_decode = torch.stack(mask_list)
        mask_decode = mask_decode[:, :num_rerank, :]

        return mask_decode.float()
