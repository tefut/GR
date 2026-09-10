import abc
import math
from typing import Dict, Tuple, List, Optional
import logging
import torch
import torch.nn as nn 
import torch_npu
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from modeling.generic.sequential.rab_modules import RABModule
from modeling.generic.sequential.utils import handle_padded_qk
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.transforers import Transformer
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
from modeling import HAS_ATTN_FUSION_OPS, ENABLE_JAGGED_OPS
from modeling.generic.utils.jagged_utils import dense_to_jagged, jagged_to_padded_dense
from modeling.generic.utils.hstu_dense_utils import hstu_dense


TransformerCacheState = Const.TransformerCacheState
# ------------------------------
# 带Latent Reasoning的Transformer实现
# ------------------------------


@ModelRegistry.register(req_hp=True)
class TransformerWithLR(Transformer):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        
        # 初始化 token_per_item（用于rel_attention_mask处理）
        feat_conf = common_hp.get("feature_conf")
        fuse_ia = feat_conf.get("fuse_ia", False)
        self.token_per_item = 2 if not fuse_ia else 1
        
        # 新增：OnePiece块级推理配置
        reasoning_cfg = model_cfg[Const.HP].get("reasoning", {})
        self.max_reason_steps: int = reasoning_cfg.get("max_reason_steps", 3)
        print("=" * 70)
        print("max_reason_steps: {}".format(self.max_reason_steps))
        print("=" * 70)
        self.task_type: str = reasoning_cfg.get("task_type", "ranking")
        
        model_hp = model_cfg.get(Const.HP, {})
        print(f'HSTU model_hp: {model_hp}')
        self.ffn_type = model_hp.get('ffn_type', None) # "ffn" or "glu_ffn" or None 
        self.ffn_expand = model_hp.get('ffn_expand', 6)
        self.gated_attn = model_hp.get("gated_attn", False)
        data_conf = common_hp.get("data_loader_conf")
        self.history_len = data_conf.get("history_length", 150)

        # FFN module initialization (same as HSTU)
        if self.ffn_type is not None: 
            self.norm_ffn = RMSNorm_npu(self._embedding_dim, eps=self._eps)
            if self.ffn_type == 'ffn':
                self.feed_forward = FeedForward(
                    dim=self._embedding_dim,
                    hidden_dim=int(self._embedding_dim * self.ffn_expand),
                    dropout=self._dropout_ratio,
                )
            elif self.ffn_type == 'glu_ffn':
                self.feed_forward = GLUFFN(
                    input_dim=self._embedding_dim, 
                    hidden_dim=self._embedding_dim, 
                    output_dim=self._embedding_dim, 
                    ffn_dim_multiplier=self.ffn_expand
                )
            else:
                raise ValueError('ffn_type must be chosen in ["ffn", "glu_ffn"]')

        # 新增：推理位置嵌入（RPE）
        if self.max_reason_steps > 0:
            self.rpe_embedding = nn.Embedding(self.max_reason_steps, self._embedding_dim)
            # 初始化RPE权重
            nn.init.normal_(self.rpe_embedding.weight, mean=0, std=0.02)

            self.register_buffer(
                "rpe_positions",
                torch.arange(self.max_reason_steps, dtype=torch.long),
                persistent=False  # 不保存到模型权重，轻量化
            )

        # LoopCTR-style deployment: multi-step train, zero-step infer.
        self.infer_zero_loop: bool = reasoning_cfg.get("infer_zero_loop", False)
        # One-time warning flags — avoid logging.warning on every forward call.
        self._warned_fusion: bool = False
        self._warned_jagged: bool = False
        # Avoids re-running torch.arange + broadcast comparison on every call.
        self._mask_structural_cache: dict = {}
        self.reasoning_activation_checkpoint: bool = reasoning_cfg.get("activation_checkpoint_reasoning", True)

        

    def debug_str(self) -> str:
        return (f"TransformerWithLR: embedding_dim={self._embedding_dim}, num_heads={self._num_heads}, "
                f"max_reason_steps={self.max_reason_steps}, block_size={self.block_size}, task_type={self.task_type}, "
                f"jagged_enabled={self.jagged_enabled}")

    def _rope_rotate(self, x, seq_len, all_timestamps, num_rerank, device):
        """
        对 Q 或 K 应用 RoPE 旋转编码
        Args:
            x: 输入张量，形状 [batch_size, seq_len, num_heads * head_dim]
            seq_len: 序列长度
            effective_length: 有效序列长度
            num_rerank: 需要推理的长度
            device: 设备（CPU/GPU）
        Returns:
            rotated_x: 旋转后的张量，形状不变
        """
        batch_size = x.shape[0]
        num_heads = self._num_heads
        head_dim = self._attention_dim
        all_effective_count = all_timestamps.bool().sum(dim=1).long()
        hist_effective_lengths = all_timestamps[:, :seq_len-num_rerank-1].bool().sum(dim=1).long()
        cand_effective_lengths = all_effective_count - hist_effective_lengths + self.history_len
        # 生成位置索引（0到seq_len-1）
        # 生成每个样本的有效位置索引（padding 位置设为0，不参与旋转）
        pos = torch.zeros((batch_size, seq_len), device=device)  # [1, seq_len]
        pos_valid = torch.arange(seq_len-num_rerank, device=device) + 1 # [1, hist_len+1]
        for i in range(batch_size):
            pos[i, 1:hist_effective_lengths[i]+1] = pos_valid[:hist_effective_lengths[i]]
            # pos 形状: [batch_size, seq_len]
            pos[i, seq_len-num_rerank:cand_effective_lengths[i]+1] = pos_valid[hist_effective_lengths[i]]
        pos = pos.unsqueeze(-1) # [batch_size, seq_len, 1]

        # 生成旋转角度：theta = 10000^(-2i/d)，i为维度索引
        dim_idx = torch.arange(self.rope_dim, device=device)  # [0, 1, ..., rope_dim-1]
        theta = 1.0 / (10000 ** (2 * dim_idx / self._attention_dim * self._num_heads))  # [rope_dim]
        
        # 计算旋转角：pos * theta（每个样本的有效位置对应正确角度，padding位置为0）
        angles = pos * theta  # [batch_size, seq_len, rope_dim]
        cos_angles = torch.cos(angles)  # [batch_size, seq_len, rope_dim]
        sin_angles = torch.sin(angles)  # [batch_size, seq_len, rope_dim]
        
        
        # 分割 x 为两部分：需要旋转的部分和不变的部分
        x_rot = x[..., :self.rope_dim]  # 前 rope_dim 维度需要旋转
        x_pass = x[..., self.rope_dim:]  # 剩余维度不旋转
        
        # 旋转操作：[x1*cos - x2*sin, x1*sin + x2*cos]（实部虚部旋转）
        # 对偶数索引和奇数索引分别处理（交替旋转）
        x_rot = torch.stack([
            x_rot[..., ::2] * cos_angles[..., ::2] - x_rot[..., 1::2] * sin_angles[..., 1::2],
            x_rot[..., ::2] * sin_angles[..., ::2] + x_rot[..., 1::2] * cos_angles[..., 1::2]
        ], dim=-1).flatten(-2)  # 合并旋转后的维度
        
        # 拼接旋转部分和不变部分
        rotated_x = torch.cat([x_rot, x_pass], dim=-1)
        return rotated_x
    
    
    
    def _extract_reasoning_block(self, hidden_states: torch.Tensor, step: int, init_seq_len: int) -> torch.Tensor:
        """修改：传入原始序列长度，准确提取推理块
        支持2D (fusion_enabled=True) 和 3D (fusion_enabled=False) 的hidden_states"""
        if step == 0:
            # 初始块：从原始输入的最后M个token提取
            start_idx = max(0, init_seq_len - self.block_size)
            if hidden_states.dim() == 2:
                # fusion_enabled=True 时，shape: [total_tokens, hidden_dim]
                return hidden_states[start_idx:init_seq_len, :]
            else:
                # fusion_enabled=False 时，shape: [batch_size, seq_len, hidden_dim]
                return hidden_states[:, start_idx:init_seq_len, :]
        else:
            # 后续块：从拼接的推理块区域提取
            start_idx = init_seq_len + (step-1) * self.block_size
            end_idx = start_idx + self.block_size
            if hidden_states.dim() == 2:
                # fusion_enabled=True 时，shape: [total_tokens, hidden_dim]
                return hidden_states[start_idx:end_idx, :]
            else:
                # fusion_enabled=False 时，shape: [batch_size, seq_len, hidden_dim]
                return hidden_states[:, start_idx:end_idx, :]


    def _generate_causal_block_mask(
            self,
            combined_seq_len: int,  # 拼接后的总序列长度
            init_seq_len: int,      # 原始输入序列长度
            block_start_idx: int,   # 当前推理块在拼接序列中的起始位置
            batch_size: int,
            attn_mask: torch.Tensor,  # 原始输入的无效掩码
            prev_step_mask: Optional[torch.Tensor] = None,  # step-1 mask [B, prev_len, prev_len], 3D only
    ) -> torch.Tensor:
        """
        生成适配拼接序列的attention mask：单向逐token + 块级因果 + 无效位置屏蔽
        优化版本：使用 torch.tril 替代 torch.eye，减少内存占用
        逻辑：原始输入部分用传入的invalid_attn_mask，推理块部分默认有效（全1）

        prev_step_mask: when provided (step >= 2), the prev-block rows are copied
        directly from this mask instead of the freshly-created causal base.  This
        prevents indirect data leakage where step-k scratchpad inherits broader
        history visibility than step-(k-1) scratchpad actually had.
        """
        
        if attn_mask.dim() == 4:
            attn_mask_base = attn_mask[:, 0, :, :]
        else:
            attn_mask_base = attn_mask

        device = attn_mask_base.device
        dtype = attn_mask_base.dtype

        # 1. 基础单向逐token掩码 — cached per combined_seq_len to avoid rebuilding every call.
        current_block_end = block_start_idx + self.block_size
        _struct = self._mask_structural_cache.get(combined_seq_len)
        if _struct is None or _struct.device != device or _struct.dtype != dtype:
            row_idx = torch.arange(combined_seq_len, device=device).unsqueeze(1)
            col_idx = torch.arange(combined_seq_len, device=device).unsqueeze(0)
            _struct = (row_idx >= col_idx).to(dtype=dtype)
            self._mask_structural_cache[combined_seq_len] = _struct
        extended_attn_mask_base = _struct.unsqueeze(0).expand(batch_size, -1, -1).clone()

        # 2. 块级因果约束：屏蔽当前块之后的未来块
        if current_block_end < combined_seq_len:
            extended_attn_mask_base[:, current_block_end:, :] = 0.0  # 未来块无法关注任何位置

        # 3. 扩展并融合原始无效位置掩码
        extended_attn_mask_base[:, :init_seq_len, :init_seq_len] = attn_mask_base  # 原始输入部分的无效位置保留
        
        # 4. 当前block复制前一个block对history的mask + 保留对所有历史block中自己的可见性
        #
        # FIX (indirect leakage): when prev_step_mask is provided (step >= 2), copy
        # the prev-block rows from the *actual* previous-step mask rather than from
        # the freshly-created causal base for the current (larger) sequence.  The
        # causal base rows for prev-block positions would give full backward
        # visibility, which is broader than what those positions were actually
        # allowed to see at step-(k-1).
        prev_block_start = block_start_idx - self.block_size
        prev_block_end = block_start_idx
        if combined_seq_len > init_seq_len:  # 仅当存在推理块时执行复制
            if prev_step_mask is not None:
                # step >= 2: prev_step_mask is 3D [B, prev_combined_len, prev_combined_len]
                # prev_combined_len == block_start_idx (the previous step's total length)
                # Copy exact rows that prev-block had in the previous step mask.
                prev_block_rows = prev_step_mask[:, prev_block_start:prev_block_end, :]  # [B, M, block_start_idx]
                extended_attn_mask_base[:, block_start_idx:current_block_end, :block_start_idx] = prev_block_rows
                # Self-attention diagonal within current scratchpad: same pattern as prev-block had
                prev_block_self = prev_step_mask[:, prev_block_start:prev_block_end, \
                    prev_block_start:prev_block_end]  # [B, M, M]
                extended_attn_mask_base[:, block_start_idx:current_block_end, \
                    block_start_idx:current_block_end] = prev_block_self
            else:
                # step == 1: prev-block is within original sequence (rows N-M..N-1).
                # section 3 has already applied attn_mask_base to [:, :N, :N], so
                # reading from extended_attn_mask_base here is correct.
                prev_block_history_mask = extended_attn_mask_base[:, prev_block_start:prev_block_end, \
                    :block_start_idx].clone()
                extended_attn_mask_base[:, block_start_idx:current_block_end, \
                    :block_start_idx] = prev_block_history_mask
                prev_block_candidate_mask = extended_attn_mask_base[:, prev_block_start:prev_block_end, \
                    prev_block_start:prev_block_end].clone()
                extended_attn_mask_base[:, block_start_idx:current_block_end, \
                    block_start_idx:current_block_end] = prev_block_candidate_mask
            
        if self.fusion_enabled and HAS_ATTN_FUSION_OPS:
            extended_attn_mask = extended_attn_mask_base.unsqueeze(1).repeat(1, self._num_heads, 1, 1)
        else:
            extended_attn_mask = extended_attn_mask_base
        
        return extended_attn_mask


    def _multi_head_attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            u: torch.Tensor,
            x_offsets: torch.Tensor,
            cached_k: torch.Tensor,
            cached_q: torch.Tensor,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            cached_outputs: torch.Tensor = torch.tensor([]),
            x: torch.Tensor = torch.tensor([]),
            gate_score: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, TransformerCacheState]:
        """修改后的MHA计算，对齐参考代码逻辑"""
        # 从实际输入张量推断批大小和序列长度，而非依赖x_offsets（在推理循环中可能不准确）
        if q.dim() == 3:
            # 标准3D case: [batch_size, seq_len, num_heads * attention_dim]
            bs = q.shape[0]
            n = q.shape[1]
        elif q.dim() == 2:
            # fusion_enabled case: [total_tokens, num_heads * attention_dim]
            bs = 1  # 伪批大小
            n = q.shape[0]
        else:
            raise ValueError(f"Unexpected q dimension: {q.dim()}")
        
        # n 代表整个需要推理的序列长度
        n: int = attn_mask.shape[-1]

        
        if self._normalization == "rel_bias":
            if delta_x_offsets[0].shape[0] > 0:
                k, q = handle_padded_qk(bs, cached_k, cached_q, delta_x_offsets, k, n, q)

            rel_attention_mask = None
            if self.all_timestamps is not None and self._rel_attn_bias is not None:
                # Relative Attention Bias --> attention bias
                # reshape qk_attn:  batch x num_head x 2n x 2n  --> batch x num_head x n x 2 x n x 2
                rel_attention_mask, self.time_bias = self._rel_attn_bias(
                         self.all_timestamps, self.past_lengths, self.num_rerank,
                         self.layer_num, self.time_bias)
                # 形如 [bs, _num_heads, (n-1), (n-1)]
                rel_attention_mask = rel_attention_mask.unsqueeze(1).repeat(1, self._num_heads, 1, self.token_per_item)
                seq_tokens = n // self.token_per_item - 1
                rel_attention_mask = rel_attention_mask.view(
                    bs, self._num_heads, seq_tokens, 1, seq_tokens
                )
                rel_attention_mask = rel_attention_mask.repeat(1, 1, 1, 1, self.token_per_item)
                rel_attention_mask = rel_attention_mask.transpose(-1, -2).reshape(bs, self._num_heads, n - 1, n - 1)
                # 形如 [bs, _num_heads, n, n]，补上user
                rel_attention_mask = torch.nn.functional.pad(rel_attention_mask, (1, 0, 1, 0), 'constant', 0.0)

            if self.fusion_enabled and HAS_ATTN_FUSION_OPS:
                # Check if attn_mask already has the heads dimension
                if attn_mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                    attn_mask = attn_mask.unsqueeze(1)
                    # 训练、评估时mask做过特殊处理，无法直接使用融合算子内置的mask。 repeat至 [bs, _num_heads, n, n]
                    mask = attn_mask.repeat(1, self._num_heads, 1, 1)
                else:  # Already has [batch_size, num_heads, seq_len, seq_len]
                    mask = attn_mask
                mask_type = 3  # custom
                if self.jagged_enabled and q.dim() == 2:
                    qk_shape = (-1, self._num_heads, self._attention_dim)
                    v_shape = (-1, self._num_heads, self._linear_dim)
                    layout = "jagged"
                    seq_offset = x_offsets.cpu().tolist()
                    out_shape = (-1, self._num_heads * self._linear_dim)
                else:
                    qk_shape = (bs, n, self._num_heads, self._attention_dim)
                    v_shape = (bs, n, self._num_heads, self._linear_dim)
                    layout = "normal"
                    seq_offset = None
                    out_shape = (bs, n, self._num_heads * self._linear_dim)

                attn_output = hstu_dense(
                    q.view(qk_shape), k.view(qk_shape), v.view(v_shape), mask, rel_attention_mask, mask_type,
                    n, self.qk_attn_denominator_value, layout, seq_offset
                ).reshape(out_shape)
            else:
               # add PE here 
                pos_encoding_type = None
                if pos_encoding_type == 'rope':
                    q = self._rope_rotate(q, n, self.all_timestamps, self.block_size, q.device)  # 对Q旋转
                    k = self._rope_rotate(k, n, self.all_timestamps, self.block_size, k.device)  # 对K旋转
                
                # 形如 [B, H, N, N]
                qk_attn = torch.einsum(
                    "bnhd,bmhd->bhnm",
                    q.view(bs, n, self._num_heads, self._attention_dim),
                    k.view(bs, n, self._num_heads, self._attention_dim),
                )

                if rel_attention_mask is not None:
                    qk_attn = qk_attn + rel_attention_mask
                
                if pos_encoding_type == 'alibi':
                    alibi_bias = self._generate_alibi_bias(n, self.all_timestamps, self.num_rerank, qk_attn.device)
                    qk_attn = qk_attn + alibi_bias

                qk_attn = F.silu(qk_attn) * self.qk_attn_denominator_value
                attn_mask = attn_mask.to(qk_attn.device)
                # Check if attn_mask already has the heads dimension
                if attn_mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                    # 形如 [B, 1, N, N]
                    attn_mask = attn_mask.unsqueeze(1)
                qk_attn = qk_attn * attn_mask

                # Capture attention weights only when a downstream scratchpad policy needs them.
                if self.training and getattr(self, "_capture_attention_weights", False):
                    attention_weights = F.softmax(qk_attn, dim=-1)  # [B, H, N, N]
                    self._last_attention_weights = attention_weights

                attn_output = torch.einsum(
                    "bhnm,bmhd->bnhd",
                    qk_attn,
                    v.view(bs, n, self._num_heads, self._linear_dim)
                ).reshape(bs, n, self._num_heads * self._linear_dim)
            
            if self.gated_attn:
                gate_in = self.normed_x
                if self.elementwise:
                    gate = torch.sigmoid(self.gate_proj(gate_in)) \
                        .view(bs, n, self._num_heads, self._linear_dim)
                else:
                    gate = torch.sigmoid(self.gate_proj(gate_in)) \
                        .view(bs, n, self._num_heads) \
                        .unsqueeze(-1)
                attn_output = attn_output.view(bs, n, self._num_heads, self._linear_dim) * gate.to(attn_output.dtype)
                attn_output = attn_output.reshape(bs, n, self._num_heads * self._linear_dim)

        else:
            raise ValueError("Unknown normalization method %s", self._normalization)

        attn_output = attn_output if delta_x_offsets[0].shape[0] == 0 else attn_output[delta_x_offsets[0], :]
        
        if self.with_attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)

        o_input = u * self._norm_attn_output(attn_output)
        # x --> u k q v
        new_outputs = self._o(
            F.dropout(
                o_input,
                p=self._dropout_ratio,
                training=self.training,
            )
        ) + x

        ## HSTU引入FFN层
        if self.ffn_type:
            if new_outputs.dim() == 2:
                ffn_input = self.norm_ffn(new_outputs)
            else:
                ffn_input = self.norm_ffn(new_outputs, self.user_len, self.hist_len)
            new_outputs = self.feed_forward(ffn_input) + new_outputs

        if delta_x_offsets[0].shape[0] > 0:
            new_outputs = cached_outputs.index_copy_(dim=0, index=delta_x_offsets[0], source=new_outputs)

        if self.return_cache_states and delta_x_offsets[0].shape[0] == 0:
            v = v.contiguous()

        current_cache = (v, q, k, new_outputs)
        return new_outputs, current_cache

    
    def forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            cache: TransformerCacheState = (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = torch.tensor([])
    ) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor]:
        """
        前向传播方法, 处理输入序列并生成输出序列.

        :param x: 输入序列的特征, 形状为(\sum_i N_i, D).
        :param x_offsets: 输入序列的偏移量, 形状为(B + 1), 表示每个序列的起始位置.
        :param all_timestamps: 可选参数, 时间戳序列, 形状为(B, N).
        :param attn_mask: 无效的注意力掩码, 形状为(B, N, N), 每个元素为0或1.
        :param past_lengths: 过去序列的长度, 形状为(B,).
        :param num_rerank: 需要推理部分的长度.
        :param layer_num: 当前层的编号.
        :param delta_x_offsets: 可选参数, 形状为((B,), (B,))的偏移量, 对于元组中的第一个元素,
            每个元素在[0,x_offsets[-1])中. 对于元组中的第2个元素, 每个元素在[0,N)中.
        :param cache: 可选参数, 缓存状态, 用于存储中间结果(v, padded_q, padded_k, output).
        :param return_cache_states: 是否返回缓存状态.
        :return: 处理后的输出序列, 形状为(\sum_i N_i, D).
        """
        # n 代表整个需要推理的序列长度
        n: int = attn_mask.shape[-1]
        # Keep public args unchanged; silently fallback to non-fusion path when fused ops are unavailable.
        self.fusion_enabled = False
        self.jagged_enabled = False
        self.use_cache = False
        _j2p_info = None

        # For non-fusion runtime, convert 2D jagged input to padded 3D once before attention.
        if self.jagged_enabled and x.dim() == 2 and not self.fusion_enabled:
            offsets_cpu = x_offsets.cpu().long()
            _j2p_lengths = offsets_cpu[1:] - offsets_cpu[:-1]
            _j2p_batch_idx = torch.arange(x_offsets.shape[0] - 1).repeat_interleave(_j2p_lengths).to(x.device)
            _j2p_seq_idx = (
                torch.arange(int(offsets_cpu[-1].item()))
                - offsets_cpu[:-1].repeat_interleave(_j2p_lengths)
            ).to(x.device)

            def _jagged_to_padded_entry(tensor_2d: torch.Tensor) -> torch.Tensor:
                d = tensor_2d.shape[-1]
                padded = torch.zeros(x_offsets.shape[0] - 1, n, d, device=tensor_2d.device, dtype=tensor_2d.dtype)
                padded[_j2p_batch_idx, _j2p_seq_idx] = tensor_2d
                return padded

            x = _jagged_to_padded_entry(x)
            _j2p_info = (_j2p_batch_idx, _j2p_seq_idx)

        if delta_x_offsets[0].shape[0] > 0 and self.use_cache:
            if cache[0][0].shape[0] == 0:
                raise ValueError("cache must be provided when delta_x_offsets is not None")
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = cache
        else:
            _et = torch.tensor([])
            cached_v = cached_q = cached_k = cached_outputs = _et
        
        batch_size: int = x_offsets.shape[0] - 1
        init_seq_len = n
        normed_x = self._norm_input(x)
        self.normed_x = normed_x
        self.return_cache_states = return_cache_states
        self.block_size = num_rerank
        self.num_rerank = num_rerank
        self.layer_num = layer_num
        self.past_lengths = past_lengths
        self.time_bias = time_bias
        self.all_timestamps = all_timestamps

        if self._linear_config == "uvqk":
            if self.with_attn_output_gate:
                u, v, q, gate_score, k = self._linear_transform(normed_x)
            else:
                u, v, q, k = self._linear_transform(normed_x)
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)


        if delta_x_offsets[0].shape[0] > 0:
            v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        attn_mask_3d = attn_mask[:, 0, :, :] if attn_mask.dim() == 4 else attn_mask
        combined_seq_len = init_seq_len

        init_outputs, init_cache = self._multi_head_attention(
            q=q, k=k, v=v, attn_mask=attn_mask_3d, u=u,
            x_offsets=x_offsets,
            cached_k=cached_k,
            cached_q=cached_q,
            delta_x_offsets=delta_x_offsets,
            cached_outputs=cached_outputs,
            x=normed_x,
            gate_score=gate_score if self.with_attn_output_gate else None
        )
        

        hidden_states = init_outputs # 初始隐藏态：[batch_size, init_seq_len, hidden_dim]
        current_cache = init_cache

        # LoopCTR infer_zero_loop: at inference time skip reasoning loop entirely
        # → latency matches plain HSTU single-pass.
        if not self.training and self.infer_zero_loop:
            if _j2p_info is not None:
                _j2p_batch_idx, _j2p_seq_idx = _j2p_info
                init_outputs = init_outputs[_j2p_batch_idx, _j2p_seq_idx]
            return init_outputs, current_cache, time_bias

        # ---- Convert jagged 2D → padded 3D for reasoning loop ----
        if _j2p_info is None and self.jagged_enabled and hidden_states.dim() == 2:
            offsets_cpu = x_offsets.cpu().long()
            _j2p_lengths = (offsets_cpu[1:] - offsets_cpu[:-1])
            _j2p_batch_idx = torch.arange(batch_size).repeat_interleave(_j2p_lengths).to(hidden_states.device)
            _j2p_seq_idx = (torch.arange(int(offsets_cpu[-1].item())) - \
                offsets_cpu[:-1].repeat_interleave(_j2p_lengths)).to(hidden_states.device)

            def _jagged_to_padded(t):
                D = t.shape[-1]
                p = torch.zeros(batch_size, init_seq_len, D, device=t.device, dtype=t.dtype)
                p[_j2p_batch_idx, _j2p_seq_idx] = t
                return p

            hidden_states = _jagged_to_padded(hidden_states)
            x = _jagged_to_padded(x)
            _j2p_info = (_j2p_batch_idx, _j2p_seq_idx)

        # 提取初始推理块（第0步）
        all_reason_blocks = []
        reason_block = self._extract_reasoning_block(hidden_states, step=0, init_seq_len=init_seq_len)
        all_reason_blocks.append(reason_block)

        # Track the 3D mask from the previous reasoning step so that
        # _generate_causal_block_mask can copy exact prev-block rows (fix for
        # indirect data leakage when step >= 2).
        prev_mask_3d: Optional[torch.Tensor] = None  # None → step 1 uses original-seq path (correct)

        # ------------------------------
        # 步骤2：多步推理迭代（1~max_reason_steps）
        # ------------------------------
        for step in range(1, self.max_reason_steps + 1):
            # 2.1 推理块添加RPE
            # forward中直接使用预创建的索引
            rpe = self.rpe_embedding(self.rpe_positions[step-1]).to(x.device) 

            if hidden_states.dim() == 2:
                rpe = rpe.unsqueeze(0).repeat(self.block_size, 1)  # [block_size, hidden_dim]
                enhanced_block = reason_block + rpe  # [block_size, hidden_dim]
                
                # 2.2 拼接输入序列 + 历史推理块 (concatenate along dim=0)
                combined_seq = torch.cat([x] + all_reason_blocks[:-1] + \
                    [enhanced_block], dim=0)  # [total_combined_tokens, hidden_dim]
                combined_seq_len = init_seq_len + step * self.block_size  # 拼接后的总长度：init_seq_len + step * block_size
            else:
                rpe = rpe.unsqueeze(0).unsqueeze(1).expand(batch_size, \
                    self.block_size, -1)  # [batch_size, block_size, hidden_dim]
                enhanced_block = reason_block + rpe  # [batch_size, block_size, hidden_dim]
                
                # 2.2 拼接输入序列 + 历史推理块 (concatenate along dim=1)
                combined_seq = torch.cat([x] + all_reason_blocks[:-1] + \
                    [enhanced_block], dim=1)  # [batch_size, combined_seq_len, hidden_dim]
                combined_seq_len = init_seq_len + step * self.block_size  # 拼接后的总长度：init_seq_len + step * block_size

            # 2.3 生成因果块掩码
            block_start_idx = init_seq_len + (step-1) * self.block_size
            current_mask = self._generate_causal_block_mask(
                combined_seq_len=combined_seq_len,
                init_seq_len=init_seq_len,
                block_start_idx=block_start_idx,
                batch_size=batch_size,
                attn_mask=attn_mask,  # 传入原始输入的无效掩码
                prev_step_mask=prev_mask_3d,  # None for step 1; actual prev mask for step >= 2
            )
            # Save 3D version for the next iteration
            prev_mask_3d = current_mask[:, 0, :, :] if current_mask.dim() == 4 else current_mask

            # 2.4 利用KV缓存优化
            if self.use_cache and current_cache[0].numel() > 0:
                cached_v, cached_q, cached_k, cached_outputs = current_cache
                # 对当前增强块做归一化和线性变换
                new_normed_x = self._norm_input(enhanced_block)
                new_linear_output = self._linear_transform(new_normed_x)
                if self.with_attn_output_gate:
                    new_u, new_v, new_q, new_gate_score, new_k = new_linear_output
                else:
                    new_u, new_v, new_q, new_k = new_linear_output
                
                # 拼接历史KV和新KV（适配拼接后的序列）
                if enhanced_block.dim() == 2:
                    concat_dim = 0
                else:
                    concat_dim = 1
                    
                k = torch.cat([cached_k, new_k], dim=concat_dim)
                v = torch.cat([cached_v, new_v], dim=concat_dim)
                u = torch.cat([u, new_u], dim=concat_dim)
                q = torch.cat([cached_q, new_q], dim=concat_dim)
                gate_score = torch.cat([gate_score, new_gate_score], \
                    dim=concat_dim) if self.with_attn_output_gate else None
            else:
                # 无缓存：对整个拼接序列做归一化和线性变换
                new_normed_x = self._norm_input(combined_seq)
                new_linear_output = self._linear_transform(new_normed_x)
                if self.with_attn_output_gate:
                    u, v, q, gate_score, k = new_linear_output
                else:
                    u, v, q, k = new_linear_output

            # 2.5 调用修改后的MHA计算
            if self.training and self.reasoning_activation_checkpoint \
                and not self.return_cache_states:
                def _lr_reasoning_step(_combined_seq: torch.Tensor, _current_mask: torch.Tensor) -> torch.Tensor:
                    _new_normed_x = self._norm_input(_combined_seq)
                    _new_linear_output = self._linear_transform(_new_normed_x)
                    if self.with_attn_output_gate:
                        _u, _v, _q, _gate_score, _k = _new_linear_output
                    else:
                        _u, _v, _q, _k = _new_linear_output
                        _gate_score = None
                    _step_outputs, _ = self._multi_head_attention(
                        q=_q, k=_k, v=_v, attn_mask=_current_mask, u=_u,
                        x_offsets=x_offsets,
                        cached_k=cached_k,
                        cached_q=cached_q,
                        delta_x_offsets=delta_x_offsets,
                        cached_outputs=cached_outputs,
                        x=_new_normed_x,
                        gate_score=_gate_score if self.with_attn_output_gate else None
                    )
                    return _step_outputs

                step_outputs = checkpoint(
                    _lr_reasoning_step,
                    combined_seq,
                    current_mask,
                    use_reentrant=False,
                )
                step_cache = (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]))
            else:
                step_outputs, step_cache = self._multi_head_attention(
                    q=q, k=k, v=v, attn_mask=current_mask, u=u,
                    x_offsets=x_offsets,
                    cached_k=cached_k,
                    cached_q=cached_q,
                    delta_x_offsets=delta_x_offsets,
                    cached_outputs=cached_outputs,
                    x=new_normed_x,  # 残差连接的输入是拼接后的序列
                    gate_score=gate_score if self.with_attn_output_gate else None
                )
            
            
            hidden_states = step_outputs  # 更新隐藏态：[batch_size, combined_seq_len, hidden_dim]
            current_cache = step_cache

            # 2.6 提取当前推理块，用于下一步迭代
            reason_block = self._extract_reasoning_block(hidden_states, step=step, init_seq_len=init_seq_len)
            all_reason_blocks.append(reason_block)


        # 最终输出：拼接序列对应的隐藏态 + 缓存 + 其他返回值
        # ------------------------------
        if hidden_states.dim() == 2:
            front_end_idx = combined_seq_len - self.block_size * (self.max_reason_steps + 1)
            new_outputs = torch.cat([hidden_states[:front_end_idx, :], hidden_states[-self.block_size:, :]], dim=0)
        else:
            front_end_idx = combined_seq_len - self.block_size * (self.max_reason_steps + 1)
            new_outputs = torch.cat([hidden_states[:, :front_end_idx, :], \
                hidden_states[:, -self.block_size:, :]], dim=1)
        
        # ---- Convert padded 3D → jagged 2D ----
        if _j2p_info is not None:
            _j2p_batch_idx, _j2p_seq_idx = _j2p_info
            new_outputs = new_outputs[_j2p_batch_idx, _j2p_seq_idx]

        if self.max_reason_steps > 0:
            return new_outputs, current_cache, time_bias
        else:
            return hidden_states, current_cache, time_bias
