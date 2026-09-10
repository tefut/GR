"""
MixFormer 核心模型模块。

从 MixFormer 参考项目迁移的核心组件：
- QueryMixer: 解耦的用户/物品 Query 混合模块
- MixFormerBlock: 单个 MixFormer 层（QueryMixer + Cross-Attention + OutputFusion）
- MixFormerModule: 多层 MixFormerBlock 的堆叠封装
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.mixformer_layers import (
    _apply_norm,
    BatchedPerHeadGLUFFN,
    HeadMixing,
)
from modeling.generic.sequential.transformers import (
    GLUFFN,
    TransformerCache,
)
from modeling.generic.utils.constants import Const
from modeling.model_registry import ModelRegistry

TransformerCacheState = Const.TransformerCacheState


class QueryMixer(nn.Module):
    """
    Query Mixer Module for multi-candidate sequence scenarios.

    Structure:
        RMSNorm -> HeadMixing -> Add -> RMSNorm -> PerHead SwiGLUFFN -> Add
    """

    def __init__(self, num_user_heads: int, num_item_heads: int, d_model: int):
        super().__init__()

        self._embedding_dim = d_model

        self.num_user_heads = num_user_heads
        self.num_item_heads = num_item_heads
        self._eps = Const.EPS

        # RMSNorm layers
        self.norm1 = RMSNorm(self._embedding_dim, eps=self._eps)
        self.norm2 = RMSNorm(self._embedding_dim, eps=self._eps)

        # Head mixing module
        self.head_mixing = HeadMixing(
            num_user_heads=self.num_user_heads,
            num_item_heads=self.num_item_heads,
            d_model=self._embedding_dim,
        )

        # Batched per-head GLUFFN for user heads
        self.user_head_ffn = BatchedPerHeadGLUFFN(
            num_heads=self.num_user_heads,
            input_dim=self._embedding_dim,
            hidden_dim=self._embedding_dim,
            output_dim=self._embedding_dim,
        )

        # Batched per-head GLUFFN for candidate heads
        self.cand_head_ffn = BatchedPerHeadGLUFFN(
            num_heads=self.num_item_heads,
            input_dim=self._embedding_dim,
            hidden_dim=self._embedding_dim,
            output_dim=self._embedding_dim,
        )

    def forward(
            self,
            user_feat: torch.Tensor,
            item_feat_seq: torch.Tensor,
            candidate_offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            user_feat: (B, 1, n_u * D)
            item_feat_seq: (B, C, n_c * D)
            candidate_offsets: Not used in dense path, kept for interface compat.

        Returns:
            user_output: (B, 1, n_u * D)
            item_output: (B, C, n_c * D)
        """
        # RMSNorm -> HeadMixing -> Add
        user_norm = _apply_norm(
            user_feat, self.norm1, self.num_user_heads, self._embedding_dim
        )
        item_norm = _apply_norm(
            item_feat_seq, self.norm1, self.num_item_heads, self._embedding_dim
        )

        user_mixed, item_mixed = self.head_mixing(
            user_norm, item_norm, candidate_offsets
        )
        user_residual = user_mixed + user_feat.view(*user_mixed.shape)
        item_residual = item_mixed + item_feat_seq.view(*item_mixed.shape)

        # RMSNorm -> PerHead SwiGLUFFN -> Add
        user_norm2 = _apply_norm(
            user_residual, self.norm2, self.num_user_heads, self._embedding_dim
        )
        item_norm2 = _apply_norm(
            item_residual, self.norm2, self.num_item_heads, self._embedding_dim
        )

        user_ffn_output = self.user_head_ffn(user_norm2)
        item_ffn_output = self.cand_head_ffn(item_norm2)

        user_output = user_ffn_output + user_residual
        item_output = item_ffn_output + item_residual

        return user_output, item_output


@ModelRegistry.register(req_hp=True, opt_subs={"QueryMixer"})
class MixFormerBlock(BaseModel):
    """
    MixFormer Block with Query Mixer, Cross Attention, and Output Fusion.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        model_conf = common_hp["model_conf"]
        seq_cfg = model_cfg[Const.HP]
        dataloader_conf = common_hp["data_loader_conf"]

        self.history_len = dataloader_conf.get("history_length", 150)
        self._embedding_dim = model_conf.get("item_embedding_dim", 128)
        if seq_cfg.get("item_embedding_dim", None):
            self._embedding_dim = seq_cfg.get("item_embedding_dim")

        self.feature_conf = common_hp.get("feature_conf", {})
        self.num_user_heads = self.feature_conf.get("n_u", 1)
        self.num_item_heads = self.feature_conf.get("n_c", 3)
        self.num_total_heads = self.num_user_heads + self.num_item_heads
        self._eps = seq_cfg.get("eps", Const.EPS)

        self.query_mixer = QueryMixer(
            self.num_user_heads, self.num_item_heads, self._embedding_dim
        )

        self.seq_norm = RMSNorm(self._embedding_dim, eps=self._eps)
        self.seq_ffn = GLUFFN(
            input_dim=self._embedding_dim,
            hidden_dim=self._embedding_dim,
            output_dim=self._embedding_dim,
        )
        self.seq_after_norm = RMSNorm(self._embedding_dim, eps=self._eps)

        self.cross_attn_norm = RMSNorm(self._embedding_dim, eps=self._eps)

        self.w_k = nn.Linear(self._embedding_dim, self._embedding_dim, bias=False)
        self.w_v = nn.Linear(self._embedding_dim, self._embedding_dim, bias=False)

        self.fusion_norm = RMSNorm(self._embedding_dim, eps=self._eps)

        self.output_fusion_u = BatchedPerHeadGLUFFN(
            num_heads=self.num_user_heads,
            input_dim=self._embedding_dim,
            hidden_dim=self._embedding_dim,
            output_dim=self._embedding_dim,
        )
        self.output_fusion_i = BatchedPerHeadGLUFFN(
            num_heads=self.num_item_heads,
            input_dim=self._embedding_dim,
            hidden_dim=self._embedding_dim,
            output_dim=self._embedding_dim,
        )

    def forward(
            self,
            user_feat: torch.Tensor,
            item_feat_seq: torch.Tensor,
            seq_feat: torch.Tensor,
            seq_offsets: torch.Tensor,
            item_offsets: torch.Tensor,
            num_rerank: int,
            all_timestamps: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            delta_seq_offsets: Optional[torch.Tensor] = None,
            return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        # ==========================================
        # 0. Basic shapes
        # ==========================================
        B = user_feat.shape[0]
        D = self._embedding_dim
        L_kv = seq_feat.shape[1]
        total_seq_len = B * L_kv

        has_delta = False
        delta_idx = None
        if delta_seq_offsets is not None:
            if isinstance(delta_seq_offsets, (tuple, list)):
                delta_idx = delta_seq_offsets[0]
            else:
                delta_idx = delta_seq_offsets
            has_delta = delta_idx is not None and delta_idx.numel() > 0

        # ==========================================
        # 2. History FFN + KV generation
        # ==========================================
        if has_delta:
            if cache is None:
                raise ValueError("cache must be provided when delta_seq_offsets is not empty")

            cached_k, cached_v = cache

            seq_feat_flat = seq_feat.reshape(total_seq_len, D)
            seq_feat_delta = seq_feat_flat.index_select(0, delta_idx)

            seq_h_delta = self.seq_after_norm(
                self.seq_ffn(self.seq_norm(seq_feat_delta)) + seq_feat_delta
            )

            k_delta = self.w_k(seq_h_delta)
            v_delta = self.w_v(seq_h_delta)

            k_flat = cached_k.index_copy(0, delta_idx, k_delta)
            v_flat = cached_v.index_copy(0, delta_idx, v_delta)

            k = k_flat.reshape(B, L_kv, D)
            v = v_flat.reshape(B, L_kv, D)

            seq_h = self.seq_after_norm(
                self.seq_ffn(self.seq_norm(seq_feat)) + seq_feat
            )
        else:
            seq_h = self.seq_after_norm(
                self.seq_ffn(self.seq_norm(seq_feat)) + seq_feat
            )

            k = self.w_k(seq_h)
            v = self.w_v(seq_h)

            k_flat = k.reshape(total_seq_len, D)
            v_flat = v.reshape(total_seq_len, D)

        new_cache = (k_flat.contiguous(), v_flat.contiguous()) if return_cache_states else None

        # ==========================================
        # 3. Generate Query via QueryMixer
        # ==========================================
        user_q, item_q_seq = self.query_mixer(
            user_feat,
            item_feat_seq,
            candidate_offsets=item_offsets,
        )
        # user_q shape: [B, 1, num_u_head, D]
        # item_q_seq shape: [B, M, num_i_head, D]

        q_u_norm = _apply_norm(
            user_q,
            self.cross_attn_norm,
            self.num_user_heads,
            D,
        )

        q_i_norm = _apply_norm(
            item_q_seq,
            self.cross_attn_norm,
            self.num_item_heads,
            D,
        )

        L_q_i = q_i_norm.shape[1]

        # ==========================================
        # 4. Build stable mask (history validity)
        # ==========================================
        hist_ts = all_timestamps[:, :L_kv]
        kv_valid_mask = hist_ts > 0

        # 如果某个样本 history 全 padding，则临时打开第 0 个 key，
        # 避免 SDPA 看到全 False mask 产生 NaN。
        #
        # 等价逻辑：
        #
        # 用 cat 写法，避免 in-place bool scatter 对导出不友好。
        has_kv = kv_valid_mask.any(dim=1)

        first_kv = kv_valid_mask[:, :1] | (~has_kv).view(B, 1)

        if L_kv > 1:
            kv_valid_mask_safe = torch.cat(
                [first_kv, kv_valid_mask[:, 1:]],
                dim=1,
            )
        else:
            kv_valid_mask_safe = first_kv

        kv_mask_safe = kv_valid_mask_safe.view(B, 1, 1, L_kv)

        query_ts = all_timestamps[:, self.history_len:self.history_len + L_q_i]
        q_i_valid_mask = query_ts > 0

        has_kv_f = has_kv.view(B, 1, 1, 1).to(dtype=q_i_norm.dtype)
        q_i_valid_f = q_i_valid_mask.view(B, 1, L_q_i, 1).to(dtype=q_i_norm.dtype)

        # ==========================================
        # 5. KV expand helper
        # ==========================================
        def expand_kv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
            return (
                x.reshape(B, L_kv, 1, D)
                .transpose(1, 2)
                .expand(B, num_heads, L_kv, D)
                .contiguous()
            )

        # ==========================================
        # 6. User Cross-Attention
        # ==========================================
        q_u = (
            q_u_norm
            .reshape(B, 1, self.num_user_heads, D)
            .transpose(1, 2)
        )

        attn_u = F.scaled_dot_product_attention(
            q_u,
            expand_kv(k, self.num_user_heads),
            expand_kv(v, self.num_user_heads),
            attn_mask=kv_mask_safe.expand(B, 1, 1, L_kv),
            dropout_p=0.0,
        )

        attn_u = attn_u * has_kv_f

        attn_u = (
            attn_u
            .transpose(1, 2)
            .reshape(B, 1, self.num_user_heads, D)
        )

        # ==========================================
        # 7. Item Cross-Attention
        # ==========================================
        q_i = (
            q_i_norm
            .reshape(B, L_q_i, self.num_item_heads, D)
            .transpose(1, 2)
        )

        attn_i = F.scaled_dot_product_attention(
            q_i,
            expand_kv(k, self.num_item_heads),
            expand_kv(v, self.num_item_heads),
            attn_mask=kv_mask_safe.expand(B, 1, L_q_i, L_kv),
            dropout_p=0.0,
        )

        attn_i = attn_i * has_kv_f * q_i_valid_f

        attn_i = attn_i.transpose(1, 2)

        # ==========================================
        # 8. Residual fusion & output
        # ==========================================
        user_out = user_q + attn_u
        item_out_seq = item_q_seq + attn_i

        user_out_norm = _apply_norm(
            user_out,
            self.fusion_norm,
            self.num_user_heads,
            D,
        )

        item_out_seq_norm = _apply_norm(
            item_out_seq,
            self.fusion_norm,
            self.num_item_heads,
            D,
        )

        user_out = user_out + self.output_fusion_u(user_out_norm)
        item_out_seq = item_out_seq + self.output_fusion_i(item_out_seq_norm)

        return user_out, item_out_seq, seq_h, new_cache


@ModelRegistry.register(req_hp=True, req_subs={"MixFormerBlock"})
class MixFormerModule(BaseModel):
    """
    MixFormer Module wrapping multiple MixFormerBlock layers.

    Expects input dict with keys 'user', 'history', 'candidate' and
    returns a processed dict with the same keys.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        self.feature_conf = common_hp.get("feature_conf", {})
        self.dataloader_conf = common_hp.get("data_loader_conf", {})
        self.model_conf = common_hp.get("model_conf", {})
        self.num_blocks = model_cfg[Const.HP].get("num_blocks", 8)

        modules = [self.init_sub_model("MixFormerBlock") for _ in range(self.num_blocks)]
        self._transformer = torch.nn.ModuleList(modules)

    def forward(
            self,
            x: Dict[str, torch.Tensor],
            x_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            cache: Optional[List[TransformerCacheState]] = None,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            **config,
    ) -> Tuple[Dict[str, torch.Tensor], TransformerCacheState]:
        required_keys = {'user', 'history', 'candidate'}
        if not required_keys.issubset(x.keys()):
            raise ValueError(
                f"Input dictionary must contain keys: {required_keys}. Got: {x.keys()}"
            )

        user_feat = x['user']
        history_feat = x['history']
        candidate_feat = x['candidate']

        B, C, n_c, D = candidate_feat.shape
        candidate_feat = candidate_feat.reshape(B, C, n_c * D)

        cache_states = config.get("cache_states") or TransformerCache()

        for _, layer in enumerate(self._transformer):
            user_out, item_out_seq, history_out, cache_state = layer(
                user_feat=user_feat,
                item_feat_seq=candidate_feat,
                seq_feat=history_feat,
                seq_offsets=x.get('history_offsets'),
                item_offsets=x.get('candidate_offsets'),
                num_rerank=num_rerank,
                all_timestamps=all_timestamps,
                attn_mask=attn_mask,
                cache=cache_states,
                delta_seq_offsets=delta_x_offsets,
                return_cache_states=return_cache_states,
            )
            user_feat = user_out
            history_feat = history_out
            candidate_feat = item_out_seq

            if return_cache_states:
                cache_states.append(cache_state)

        new_outputs = {
            'user': user_feat,
            'history': history_feat,
            'candidate': candidate_feat.view(B, C, -1),
        }
        return new_outputs, cache_states
