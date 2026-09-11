from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.utils.constants import Const, FeatConst
from modeling.model_registry import ModelRegistry


@ModelRegistry.register(multi_sel_multi_subs=[{"CausalAttentionMask", "TimeAttentionMask", "MTAttentionMask"}])
class AttentionMaskModule(BaseModel):
    """
    "AttentionMaskModule": {
        "type": ["CausalAttentionMask", "TimeAttentionMask"],
        "cfg": {}
    }

    :param model_conf:
    :param model_factory:
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:

        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp.get("model_conf", {})
        train_conf = common_hp.get("train_conf", {})
        self._bf16_mode = (train_conf.get("use_amp", False) and
                           model_conf.get("amp_dtype", "fp16") == "bf16")
        self.attention_mask_modules = nn.ModuleList([
            self.init_sub_model(sub_key)
            for sub_key in model_cfg[Const.SUB_MODELS].keys()
        ])

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank) -> torch.Tensor:
        init_mask = None
        for attn_module in self.attention_mask_modules:
            mask = attn_module(model_inputs=model_inputs, max_seq_len=max_seq_len, num_rerank=num_rerank)
            if init_mask is None:
                init_mask = mask
            else:
                init_mask = init_mask * mask
        attn_mask = init_mask.detach().clone()

        return attn_mask


@ModelRegistry.register()
class CausalAttentionMask(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        因果注意力掩码生成
        训练时：生成形状为(bs, 2 * max_seq_len + 2, 2 * max_seq_len + 2)的下三角矩阵
        推理时：生成形状为(bs, 2 * max_seq_len + 2 + num_rerank, 2 * max_seq_len + 2 + num_rerank)的掩码矩阵，
        其中前2 * max_seq_len + 2行/列与训练时的掩码矩阵相同，但候选集部分token互相不可见
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp.get("model_conf", {})
        train_conf = common_hp.get("train_conf", {})
        self._bf16_mode = (train_conf.get("use_amp", False) and
                           model_conf.get("amp_dtype", "fp16") == "bf16")
        feat_conf = common_hp.get("feature_conf")
        self.mask_candidates = model_cfg[Const.HP].get("mask_candidates", False)

        fuse_ia = feat_conf.get("fuse_ia", False)
        self.token_per_item = 2 if not fuse_ia else 1

        self.hist_dates_key = feat_conf.get("history_date_column", FeatConst.DFLT_HIST_DATE_KEY)
        self.cand_dates_key = feat_conf.get("candidate_date_column", FeatConst.DFLT_CAND_DATE_KEY)

        # 掩码模板缓存：key=(hist_len, cand_len), value=mask tensor (不含batch维度)
        self._mask_cache = {}

    def init_mask_for_export(self, seq_len, num_rerank, device):
        max_len = seq_len * self.token_per_item + 1 + num_rerank
        _pos_indices = torch.arange(max_len).repeat(max_len).view(max_len, max_len).to(device)
        mask_dtype = torch.bfloat16 if self._bf16_mode else torch.float32
        _base_mask = (_pos_indices.t() > _pos_indices).to(mask_dtype)
        _identity = (_pos_indices.t() == _pos_indices).to(mask_dtype)
        return _pos_indices, _identity, _base_mask

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank):
        if self.hist_dates_key in model_inputs.keys():
            device = model_inputs[self.hist_dates_key].device
        else:
            device = model_inputs["action_dates_key"].device
        max_len = max_seq_len * self.token_per_item + num_rerank + 1
        if not self.mask_candidates:
            if self.hist_dates_key in model_inputs.keys():
                hist_ts = model_inputs[self.hist_dates_key]
                cand_ts = model_inputs[self.cand_dates_key]
            else:
                hist_ts = model_inputs["action_dates_key"]
                cand_ts = model_inputs["cand_dates_key"]
            bs, hist_len = hist_ts.shape
            _, cand_len = cand_ts.shape
            device = hist_ts.device
            # 计算总长度：user(1) + history + candidate
            total_len = 1 + hist_len + cand_len

            # 尝试从缓存获取 mask 模板
            cache_key = (hist_len, cand_len, self._bf16_mode)
            mask_template = self._mask_cache.get(cache_key)

            if mask_template is not None and mask_template.shape[0] == total_len:
                # 缓存命中：直接 expand 到 batch 维度
                attn_mask = mask_template.unsqueeze(0).expand(bs, -1, -1)
                return attn_mask

            # 缓存未命中：构建 mask 并缓存
            # 创建基础的掩码
            mask = torch.zeros(total_len, total_len, device=device)

            # 历史和candidate都可见user
            mask[:, 0] = 1

            # 历史之间causal
            hist_indices = torch.arange(hist_len, device=device)
            hist_causal = hist_indices.unsqueeze(0) <= hist_indices.unsqueeze(1)
            # 第一步：定义关键边界
            row_start, row_end = 1, hist_len + 1
            col_start, col_end = 1, hist_len + 1

            # 第二步：拆分 mask 为多个部分
            mask_top = mask[:row_start, :]
            mask_bottom = mask[row_end:, :]

            # 2. 列方向拆分：中间行区域的左列、右列
            mask_mid_left = mask[row_start:row_end, :col_start]
            mask_mid_right = mask[row_start:row_end, col_end:]

            # 第三步：处理 hist_causal 并拼接中间行
            mask_dtype = torch.bfloat16 if self._bf16_mode else torch.float32
            hist_causal_float = hist_causal.to(mask_dtype)
            mask_mid_new = torch.cat([mask_mid_left, hist_causal_float, mask_mid_right], dim=1)

            # 第四步：拼接所有行得到最终结果
            mask = torch.cat([mask_top, mask_mid_new, mask_bottom], dim=0)
            # 候选可见自己和所有历史
            cand_start = hist_len + 1
            if cand_len > 0:
                mask.narrow(0, cand_start, mask.size(0) - cand_start) \
                    .narrow(1, 0, hist_len + 1) \
                    .fill_(1)
                mask[torch.arange(cand_start, total_len, device=device),
                torch.arange(cand_start, total_len, device=device)] = 1

            # 缓存 mask 模板（不含batch维度）
            if len(self._mask_cache) < 32:
                self._mask_cache[cache_key] = mask.detach().clone()

            # 扩展到batch维度
            attn_mask = mask.unsqueeze(0).expand(bs, -1, -1)
        else:
            if num_rerank == 0:
                max_len = max_seq_len * self.token_per_item + 1
                indices = torch.arange(max_len).to(device)
                t = indices.expand(max_len, max_len)
                attn_mask = (t.t() >= indices).unsqueeze(0)
            else:
                past_lengths = model_inputs['past_lengths']
                past_lengths = 1 + past_lengths * self.token_per_item
                _past_lengths = past_lengths.unsqueeze(-1).unsqueeze(-1)
                _pos_indices, _identity, _base_mask = self.init_mask_for_export(
                    seq_len=max_seq_len, num_rerank=num_rerank, device=device
                )
                seq_mask = (_pos_indices < _past_lengths).int()
                attn_mask = (seq_mask * _base_mask + _identity)

        return attn_mask


@ModelRegistry.register()
class PastCausalAttentionMask(CausalAttentionMask):

    def forward(self, model_inputs, max_seq_len, num_rerank):
        """
        Returns a bs x (2 * max_seq_len + 2) x (2 * max_seq_len + 2) attn mask
        - [1:idx+1, 1:idx+1] 全可见
        - [idx+1:, idx+1:] 下三角（含对角线）
        where idx = non_zero_index per sample.
        """

        device = model_inputs['past_end_index'].device
        max_len = max_seq_len * self.token_per_item + 2 + num_rerank

        # 每行第一个 1 的位置，即1y-30d这一段；全 0 时应设为 max_len-1 或提前处理为 max_len-1
        past_end_index = model_inputs['past_end_index'] * self.token_per_item

        bs = past_end_index.shape[0]

        row_idx = torch.arange(max_len, device=device).view(1, max_len, 1).expand(bs, -1, -1)
        col_idx = torch.arange(max_len, device=device).view(1, 1, max_len).expand(bs, -1, -1)

        nz = past_end_index.view(bs, 1, 1)

        # 前半部分：row, col 都 ≤ nz，全可见
        pre_mask = (row_idx <= nz) & (col_idx <= nz)

        # 后半部分：row, col 都 ≥ nz+1 且 row ≥ col （下三角），除了最后num_rerank个
        r_start = max_len - num_rerank
        post_mask = (row_idx > nz) & (row_idx < r_start) & (row_idx >= col_idx)

        # 最后num_rerank：能看见自己以及之前num_rerank之前所有的
        rerank_mask = (row_idx >= r_start) & (
                (col_idx < r_start) |
                (row_idx == col_idx)
        )

        # 合并并返回 float mask
        mask_dtype = torch.bfloat16 if self._bf16_mode else torch.float32
        attn_mask = (pre_mask | post_mask | rerank_mask).to(mask_dtype)

        attn_mask[:, 0, :] = 0.
        attn_mask[:, 0, 0] = 1.

        return attn_mask


@ModelRegistry.register()
class TimeAttentionMask(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        时间注意力掩码生成
        生成形状为(bs, 2 * max_seq_len + 2 + num_rerank, 2 * max_seq_len + 2 + num_rerank)的掩码矩阵，
        其中每个token的可见范围由timestamp_mask_threshold确定。
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp.get("model_conf", {})
        train_conf = common_hp.get("train_conf", {})
        self._bf16_mode = (train_conf.get("use_amp", False) and
                           model_conf.get("amp_dtype", "fp16") == "bf16")
        feat_conf = common_hp.get("feature_conf")
        self._time_threshold = model_cfg[Const.HP].get('timestamp_mask_threshold', 86400)
        self._infer_items_key = feat_conf.get('infer_items_key', 'item_id')
        self._infer_timestamps_key = feat_conf.get("infer_timestamps_key", "timestamps")

        fuse_ia = feat_conf.get("fuse_ia", False)
        self.token_per_item = 2 if not fuse_ia else 1

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank):
        device = model_inputs[self.candidate_timestamps_key].device
        bs = model_inputs.get(self.candidate_items_key).shape[0]
        all_timestamps = model_inputs[self._infer_timestamps_key].detach()

        mask_dtype = torch.bfloat16 if self._bf16_mode else torch.float32
        time_threshold_mask = (
                (all_timestamps.unsqueeze(2) - all_timestamps.unsqueeze(1)) <= self._time_threshold
        ).to(mask_dtype)

        # 对角线置0
        _pos_indices = torch.arange(max_seq_len).repeat(max_seq_len).view(max_seq_len, max_seq_len).to(device)
        _identity = (_pos_indices.t() == _pos_indices).to(mask_dtype)
        time_threshold_mask = time_threshold_mask - _identity

        attn_mask = (
                1.0
                - time_threshold_mask
                .unsqueeze(1).unsqueeze(-1)
                .repeat(1, 1, 1, self.token_per_item, self.token_per_item)
                .reshape(bs, max_seq_len * self.token_per_item, max_seq_len * self.token_per_item)
        )

        attn_mask = F.pad(attn_mask, (1, 1 + num_rerank, 1, 1 + num_rerank), 'constant', 1.0)

        return attn_mask


@ModelRegistry.register()
class MTAttentionMask(BaseModel):
    """
    Meituan方案mask，history部分causal，candidate互相不可见，仅可看见history时间位于其前面的item
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp.get("model_conf", {})
        train_conf = common_hp.get("train_conf", {})
        self._bf16_mode = (train_conf.get("use_amp", False) and
                           model_conf.get("amp_dtype", "fp16") == "bf16")
        feat_conf = common_hp.get("feature_conf")
        self.hist_dates_key = feat_conf.get("history_date_column", FeatConst.DFLT_HIST_DATE_KEY)
        self.cand_dates_key = feat_conf.get("candidate_date_column", FeatConst.DFLT_CAND_DATE_KEY)
        self._phase = common_hp.get("train_conf").get("phase", "pretrain")

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len: int, num_rerank: int) -> torch.Tensor:
        """
            Returns a bs x (h+p+1) x (h+p+1)  attn mask
        """
        if self._phase == "pretrain":
            # 默认candidate可以看见所有历史序列
            hist_ts = model_inputs[self.hist_dates_key]
            cand_ts = model_inputs[self.cand_dates_key]
            hist_lengths = model_inputs.get("history_lengths")
            device = hist_ts.device
            bs, hist_len = hist_ts.shape
            _, cand_len = cand_ts.shape

            # 总长度：user(1) + history(hist_len) + candidate(cand_len)
            total_len = 1 + hist_len + cand_len

            # 创建完整的注意力mask矩阵
            full_mask = torch.zeros(bs, total_len, total_len, device=device, dtype=torch.bool)

            # 1. user token可以被所有token看见 (第0行)
            full_mask[:, 0, :] = True

            # 2. history部分causal mask (上三角为False)
            for i in range(1, 1 + hist_len):
                full_mask[:, i, :i + 1] = True  # 可以看见自己和之前的所有token（包括user）

            # 3. candidate部分：可以看见user和所有历史序列，但不能看见其他candidate
            hist_indices = torch.arange(hist_len, device=device)
            hist_padding_mask = hist_indices.unsqueeze(0) < hist_lengths.unsqueeze(1)  # [bs, hist_len]

            for i in range(1 + hist_len, total_len):
                # 可以看见user token (位置0)
                full_mask[:, i, 0] = True
                # 可以看见所有非padding的历史token
                full_mask[:, i, 1:1 + hist_len] = hist_padding_mask
                # 可以看见自己
                full_mask[:, i, i] = True

            return full_mask.to(torch.bfloat16 if self._bf16_mode else torch.float32)

        else:
            hist_ts = model_inputs["action_dates_key"]
            cand_ts = model_inputs["cand_dates_key"]
            bs, hist_len = hist_ts.shape
            _, cand_len = cand_ts.shape
            max_len = hist_len + cand_len + 1
            device = hist_ts.device

            # final_mask 示意图 (1=True, 0=False)
            #
            #       |  u | h0 | h1 | h2 | h3 | p0 | p1 | p2 |
            #       -----------------------------------------
            #    u  |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
            #   h0  |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |
            #   h1  |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |
            #   h2  |  1 |  1 |  1 |  1 |  0 |  0 |  0 |  0 |
            #   h3  |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  0 |
            #   p0  |  1 |  1 |  1 |  1 |  0 |  1 |  0 |  0 |
            #   p1  |  1 |  1 |  1 |  1 |  1 |  0 |  1 |  0 |
            #   p2  |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  1 |

            user_col = torch.zeros(bs, 1, device=device, dtype=hist_ts.dtype)
            seq_ts = torch.cat([user_col, hist_ts, cand_ts], dim=1)
            pad_len = max_len - seq_ts.shape[1]
            ts_pad = F.pad(seq_ts, (0, pad_len), mode='constant', value=0)

            idx = torch.arange(max_len, device=device)
            row_idx = idx.view(1, max_len, 1)
            col_idx = idx.view(1, 1, max_len)
            ts_i = ts_pad.unsqueeze(2)
            ts_j = ts_pad.unsqueeze(1)
            valid_row = (ts_i > 0) | (row_idx == 0)
            valid_col = (ts_j > 0) | (col_idx == 0)

            # 四个区域
            # 用户列：所有行都可以看到用户信息
            user_mask = (col_idx == 0)
            # 历史区域：历史item只能看到它之前的历史（包括自己）
            hist_region = (col_idx >= 1) & (col_idx < 1 + hist_len) & (row_idx < hist_len + 1)
            hist_mask = hist_region & (row_idx >= col_idx)
            # 候选区域：候选item可以看到所有历史序列
            cand_region = (row_idx >= 1 + hist_len) & (col_idx >= 1) & (col_idx < 1 + hist_len)
            cand_hist_mask = cand_region
            # 预测对角线：候选item自己的位置
            pred_diag_mask = (row_idx == col_idx) & (row_idx >= 1 + hist_len)

            # 合并
            mask = ((user_mask | hist_mask | cand_hist_mask | pred_diag_mask) & valid_row & valid_col).to(
                torch.bfloat16 if self._bf16_mode else torch.float32)

        return mask
