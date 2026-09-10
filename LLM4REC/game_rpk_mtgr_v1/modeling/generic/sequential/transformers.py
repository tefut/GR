import abc
import math
from typing import Dict, Tuple, List, Optional
import logging
import torch
import torch_npu
import torch.nn.functional as F
from modeling.generic.sequential.rab_modules import RABModule
from modeling.generic.sequential.utils import handle_padded_qk
from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
from modeling.generic.sequential import HAS_ATTN_FUSION_OPS

TransformerCacheState = Const.TransformerCacheState


class GLUFFN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        multiple_of: int = 2,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1: Linear transformation for the first layer.
            w2: Linear transformation for the second layer.
            w3: Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, output_dim, bias=False)
        self.w3 = torch.nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerCache:
    def __init__(self, n: int = 0):
        self.cached_v = torch.tensor([])
        self.cached_q = torch.tensor([])
        self.cached_k = torch.tensor([])
        self.cached_outputs = torch.tensor([])
        self.n = 0

    def append(self, cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        向缓存中增加新的元素
        """
        v, q, k, outputs = cache
        is_same_number = all([t.shape[0] == self.n for t in (v, q, k, outputs)])
        if self.n == 0 or is_same_number:
            self.cached_v = torch.cat((self.cached_v, v), dim=0)
            self.cached_q = torch.cat((self.cached_q, q), dim=0)
            self.cached_k = torch.cat((self.cached_k, k), dim=0)
            self.cached_outputs = torch.cat((self.cached_outputs, outputs), dim=0)
            self.n += 1
        else:
            raise ValueError("New elements must have the same number of caches as the current cache size.")

    def select(self, index: int = 0):
        """
        根据索引取出特定的缓存元素
        """
        if index < 0 or index >= self.n:
            raise IndexError("Index out of range.")

        return self.cached_v[index], self.cached_q[index], self.cached_k[index], self.cached_outputs[index]


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm_npu(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # BF16有与FP32相同的指数范围，x.pow(2).mean()不会溢出，无需转FP32
        # FP16下仍需FP32保护（pow(2).mean可能溢出），由autocast自动处理或显式转
        if x.dtype == torch.float16:
            output = self._norm(x.float()).type_as(x)
        else:
            # BF16/BF16-FP32: 直接在原dtype计算，避免NPU上频繁dtype转换导致的流水线停顿
            output = self._norm(x)
        return output * self.weight


class Transformer(BaseModel):
    """
    基础的 Sequential Transduction Unit, STU 用于处理序列数据.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        model_conf = common_hp.get("model_conf")
        sequential_module_config = model_cfg[Const.HP]
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 128)
        self._linear_dim: int = sequential_module_config.get("dv", 32)
        self._attention_dim: int = sequential_module_config.get("dqk", 32)
        self._num_heads: int = sequential_module_config.get("num_heads", 4)
        self._linear_config: str = sequential_module_config.get("linear_config", "uvqk")
        self._linear_activation: str = sequential_module_config.get("linear_activation", "silu")
        self._dropout_ratio: float = model_conf.get("linear_dropout_rate", 0.3)
        self._attn_dropout_ratio: float = model_conf.get("attn_dropout_rate", 0.0)
        self._normalization: str = model_conf.get("normalization", "rel_bias")
        self._rel_attn_bias: RABModule = self.init_sub_model("RABModule") if "RABModule" in \
                                                                             model_cfg[Const.SUB_MODELS] else None
        self._eps: float = Const.EPS
        self._use_dynamic_padding: bool = common_hp.get("data_loader_conf", {}).get("use_dynamic_padding", False)

        if self._linear_config == "uvqk":
            self._uvqk = torch.nn.Parameter(
                torch.empty((self._embedding_dim, self._linear_dim * 2 * self._num_heads +
                             self._attention_dim * self._num_heads * 2)).normal_(mean=0, std=0.02), )
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)

        self._o = torch.nn.Linear(in_features=self._linear_dim * self._num_heads, out_features=self._embedding_dim)
        torch.nn.init.xavier_uniform_(self._o.weight)

        self.layer_norm_input = RMSNorm_npu(self._embedding_dim, eps=self._eps)
        self.layer_norm_attn_output = RMSNorm_npu(self._linear_dim * self._num_heads, eps=self._eps)

        qk_attn_denominator = sequential_module_config.get("qk_attn_denominator", "emb_dim")
        if qk_attn_denominator == "emb_dim":
            self.qk_attn_denominator_value = 1 / self._embedding_dim
        elif qk_attn_denominator == "sqrt_d":
            self.qk_attn_denominator_value = 1 / math.sqrt(self._embedding_dim)
        elif qk_attn_denominator == "max_seq_len":
            self.qk_attn_denominator_value = 1 / (self._rel_attn_bias._max_seq_len * 2 + 2)
        else:
            raise ValueError("Unknown string %s", qk_attn_denominator)

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm_input(x)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm_attn_output(x)

    """
    线性变换层用于从原始的 x 输出 q, k, v.
    """

    def _linear_transform(self, normed_x: torch.Tensor) -> torch.Tensor:
        if self._linear_config == "uvqk":
            batched_mm_output = torch.matmul(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            # u 特征交互, qkv transformer
            u, v, q, k = torch.split(
                batched_mm_output,
                [self._linear_dim * self._num_heads, self._linear_dim * self._num_heads,
                 self._attention_dim * self._num_heads, self._attention_dim * self._num_heads],
                dim=-1,
            )
            return u, v, q, k
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            invalid_attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            cache: TransformerCacheState = (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = None,
            _x_offsets_list=None
    ) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor, torch.Tensor]:
        pass


@ModelRegistry.register()
class TimeAwareRoPE(BaseModel):
    """
    时间感知旋转位置编码（Time-aware Rotary Position Embedding）
    基于论文SynerGen的设计：直接将Unix时间戳融入RoPE，编码绝对时间和相对时间差
    核心特性：
    1. 输入为Unix时间戳（秒级），无需离散化
    2. 注意力分数仅依赖相对时间差（shift invariance）
    3. 支持不同时间粒度的桶化（bucket）以控制计算复杂度
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        继承父类Transformer的参数
        内部参数修复：固定d_model，正确初始化ToRoPE的theta，不修改输入接口
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp.get("model_conf")
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 128)
        model_hp = model_cfg.get(Const.HP, {})

        # 从配置读取固定 d_model
        d_model = model_hp.get("d_model", self._embedding_dim)
        max_time_gap = model_hp.get("max_time_gap", 31536000)
        bucket_size = model_hp.get("bucket_size", None)
        alpha = model_hp.get("alpha", 0.5)  # 论文核心参数，可配
        theta_0 = model_hp.get("theta_0", 10000.0)

        logging.info("d_model %s", d_model)

        if d_model % 2 != 0:
            raise ValueError("d_model must be even for RoPE")

        self.d_model = d_model
        self.max_time_gap = max_time_gap
        self.bucket_size = bucket_size
        self.alpha = alpha
        self.theta_0 = theta_0

        # ====================== 关键修复：正确 ToRoPE 论文 theta 初始化 ======================
        d_half = d_model // 2
        positions = torch.arange(d_half)  # 维度必须是 d_model//2
        # ToRoPE 论文公式：θ_i = θ_0 * α^(i / d_half)
        theta = theta_0 * (alpha ** (positions / d_half))
        self.register_buffer("theta", theta, persistent=False)  # shape: [d_model//2]

    def _bucketize_time(self, time: torch.Tensor) -> torch.Tensor:
        """
        修复后的安全分桶（每个序列独立、不破坏时间相对性）
        """
        if self.bucket_size is None:
            t0 = time[:, :1]
            delta_t = time - t0
            return delta_t

        # 1. 每个序列独立算偏移
        t0 = time[:, :1]
        delta_t = time - t0

        # 2. 分桶（浮点型，不转整数）
        bucketed = delta_t / self.bucket_size  # 不使用 // 除法

        # 3. 裁剪
        max_bucket = self.max_time_gap / self.bucket_size
        bucketed = torch.clamp(bucketed, 0, max_bucket)

        return bucketed  # 保持浮点，不转 long

    def forward(
            self,
            x: torch.Tensor,
            seq_len: int,
            timestamps: torch.Tensor,
            num_rerank: int,
            device: str
    ) -> torch.Tensor:
        """
        前向传播：将时间感知RoPE应用到【历史行为序列】
        历史序列：有时间戳 → 做旋转
        Candidate 序列：无时间 → 不旋转
        Args:
            x: 输入向量 [batch_size, seq_len, d_model]
                结构：[历史序列 + candidate]
            timestamps: 历史序列的时间戳 [batch_size, hist_len]
        Returns:
            rotated_x: 应用RoPE后的向量 [batch_size, seq_len, d_model]
        """
        _, _, d_model = x.shape
        hist_len = seq_len - num_rerank  # 历史序列长度
        d_half = d_model // 2

        if d_model != self.d_model:
            raise ValueError(f"输入维度 {d_model} 与模型维度 {self.d_model} 不匹配")

        # ====================== 1. 拆分：历史序列 / Candidate 序列 ======================
        x_user = x[:, :1, :]
        x_hist = x[:, 1:hist_len, :]  # 历史：需要旋转
        x_cand = x[:, hist_len:, :]  # 候选：不旋转

        # ====================== 2. 历史序列时间处理 ======================
        timestamps_bucketed = self._bucketize_time(timestamps)  # [B, hist_len]

        # ====================== 3. 计算旋转角度（仅对历史） ======================
        t = timestamps_bucketed.unsqueeze(-1).repeat(1, 1, d_half)  # [B, hist_len, d_half]
        angles = t * self.theta.unsqueeze(0).unsqueeze(0)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # ====================== 4. 对历史序列执行 ToRoPE 旋转 ======================
        x1 = x_hist[..., ::2]
        x2 = x_hist[..., 1::2]

        x_hist_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)

        # ====================== 5. 拼接：旋转后的历史 + 不变的候选 ======================
        rotated_x = torch.cat([x_user, x_hist_rotated, x_cand], dim=1)
        return rotated_x


@ModelRegistry.register(opt_subs={"RABModule", "TimeAwareRoPE"})
class HSTU(Transformer):
    """
    HSTU模型用于处理序列数据.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):

        """
        继承父类Transformer的参数
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        feat_conf = common_hp.get("feature_conf")
        fuse_ia = feat_conf.get("fuse_ia", False)
        self.token_per_item = 2 if not fuse_ia else 1

        # model_hp
        model_hp = model_cfg.get(Const.HP, {})
        self.pos_encoding_type = feat_conf.get("pos_encoding_type")
        logging.info("pos_encoding_type: %s", self.pos_encoding_type)
        self.rope_rotate = self.init_sub_model("TimeAwareRoPE")

        # ffn
        self.ffn_type = model_hp.get('ffn_type', None)  # "ffn" or "glu_ffn" or None
        self.ffn_expand = model_hp.get('ffn_expand', 6)
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

    def forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            invalid_attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            cache: TransformerCacheState = (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = torch.tensor([]),
            action_bias: torch.Tensor = None,
            _x_offsets_list=None,
            _history_length=None
    ) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor]:
        """
        前向传播方法, 处理输入序列并生成输出序列.

        :param x: 输入序列的特征, 形状为(\sum_i N_i, D).
        :param x_offsets: 输入序列的偏移量, 形状为(B + 1), 表示每个序列起始位置.
        :param all_timestamps: 可选参数, 时间戳序列, 形状为(B, N).
        :param invalid_attn_mask: 无效的注意力掩码, 形状为(B, N, N), 每个元素为0或1.
        :param past_lengths: 过去序列的长度, 形状为(B,).
        :param num_rerank: 需要推理部分的长度.
        :param layer_num: 当前层的编号.
        :param delta_x_offsets: 可选参数, 形状为((B,), (B,))的偏移量, 对于元组中的第一个元素,
            每个元素在[0,x_offsets[-1])中. 对于元组中的第2个元素, 每个元素在[0,N)中.
        :param cache: 可选参数, 缓存状态, 用于存储中间结果(v, padded_q, padded_k, output).
        :param return_cache_states: 是否返回缓存状态.
        :param time_bias: 时间偏移量.
        :param action_bias: 行为类型注意力偏置, 形状为(B, N, N), 用于PinRec风格的行为条件化.
        :param _x_offsets_list: 预计算的x_offsets.tolist()结果, 由TransformerInner传入以避免每层重复NPU→CPU同步.
            为None时在层内调用x_offsets.tolist()作为兜底.
        :param _history_length: 预计算的history_length, 由TransformerInner传入以避免每层RAB模块重复.item() NPU→CPU同步.
        :return: 处理后的输出序列, 形状为(\sum_i N_i, D).
        """
        jagged_enabled = HAS_ATTN_FUSION_OPS and num_rerank == 0 and self._normalization == "rel_bias"
        if jagged_enabled:
            x = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets])[0]
        # n 代表整个需要推理的序列长度
        n: int = invalid_attn_mask.shape[-1]
        if delta_x_offsets[0].shape[0] > 0:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # 维度 [delta_x_offsets[0], :].
            cached_v = torch.zeros_like(x, device=x.device)
            cached_q = torch.zeros_like(x, device=x.device)
            cached_k = torch.zeros_like(x, device=x.device)
            cached_outputs = torch.zeros_like(x, device=x.device)
            if cache[0][0].shape[0] == 0:
                raise ValueError("cache must be provided when delta_x_offsets is not None")
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = cache

        normed_x = self._norm_input(x)

        if self._linear_config == "uvqk":
            u, v, q, k = self._linear_transform(normed_x)
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)

        if delta_x_offsets[0].shape[0] > 0:
            v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        bs: int = x_offsets.shape[0] - 1
        if self._normalization == "rel_bias":
            if delta_x_offsets[0].shape[0] > 0:
                k, q = handle_padded_qk(bs, cached_k, cached_q, delta_x_offsets, k, n, q)

            rel_attention_mask = None
            if all_timestamps is not None and self._rel_attn_bias is not None:
                # Relative Attention Bias --> attention bias
                # reshape qk_attn:  batch x num_head x 2n x 2n  --> batch x num_head x n x 2 x n x 2
                rel_attention_mask, time_bias = self._rel_attn_bias(
                    all_timestamps, past_lengths, num_rerank,
                    layer_num, time_bias, history_length=_history_length)
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
            if self.pos_encoding_type == 'rope':
                q = self.rope_rotate(q, n, all_timestamps, num_rerank, q.device)  # 对Q旋转
                k = self.rope_rotate(k, n, all_timestamps, num_rerank, k.device)  # 对K旋转

            if HAS_ATTN_FUSION_OPS:
                if num_rerank == 0:
                    qk_shape = (-1, self._num_heads, self._attention_dim)
                    v_shape = (-1, self._num_heads, self._linear_dim)
                    mask = None
                    mask_type = 0  # 0: tril
                    layout = "jagged"
                    seq_offset = _x_offsets_list if _x_offsets_list is not None else x_offsets.tolist()
                    out_shape = (-1, self._num_heads * self._linear_dim)
                else:
                    qk_shape = (bs, n, self._num_heads, self._attention_dim)
                    v_shape = (bs, n, self._num_heads, self._linear_dim)
                    # 推理时mask做过特殊处理，无法直接使用融合算子内置的mask。 repeat至 [bs, _num_heads, n, n]
                    mask = invalid_attn_mask.unsqueeze(1)
                    mask = mask.repeat(1, self._num_heads, 1, 1)
                    mask_type = 3  # 3: custom
                    layout = "normal"
                    seq_offset = None
                    out_shape = (bs, n, self._num_heads * self._linear_dim)

                # mask_type: 0 tril, 1 triu, 2 none, 3 custom. layout: "normal" padding, "jagged" non-padding

                attn_output = torch.ops.mxrec.hstu_dense(
                    q.view(qk_shape), k.view(qk_shape), v.view(v_shape), mask, rel_attention_mask, mask_type,
                    n, self.qk_attn_denominator_value, layout, seq_offset
                ).reshape(out_shape)
            else:
                # 形如 [B, H, N, N]
                qk_attn = torch.einsum(
                    "bnhd,bmhd->bhnm",
                    q.view(bs, n, self._num_heads, self._attention_dim),
                    k.view(bs, n, self._num_heads, self._attention_dim),
                )

                if rel_attention_mask is not None:
                    qk_attn = qk_attn + rel_attention_mask
                # Add action-type attention bias (PinRec-inspired)
                if action_bias is not None:
                    # action_bias: [B, 1, N] -> [B, H, 1, N], broadcast over query rows
                    action_bias = action_bias.unsqueeze(1).expand(-1, self._num_heads, -1, -1)
                    qk_attn = qk_attn + action_bias
                qk_attn = F.silu(qk_attn) * self.qk_attn_denominator_value
                # FP16 safety: clamp before mask multiply to prevent 0*Inf=NaN
                if qk_attn.dtype == torch.float16:
                    _fp16_max = torch.finfo(qk_attn.dtype).max
                    qk_attn = torch.clamp(qk_attn, min=-_fp16_max, max=_fp16_max)
                attn_mask = invalid_attn_mask.to(qk_attn.device)
                # 形如 [B, 1, N, N]
                attn_mask = attn_mask.unsqueeze(1)
                qk_attn = qk_attn * attn_mask
                # FP16 safety: NaN from 0*Inf in masked positions → 0
                if qk_attn.dtype == torch.float16:
                    qk_attn = torch.nan_to_num(qk_attn, nan=0.0)
                attn_output = torch.einsum(
                    "bhnm,bmhd->bnhd",
                    qk_attn,
                    v.view(bs, n, self._num_heads, self._linear_dim)
                ).reshape(bs, n, self._num_heads * self._linear_dim)
                # FP16 safety: sanitize attention output
                if attn_output.dtype == torch.float16:
                    _fp16_max = torch.finfo(attn_output.dtype).max
                    attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=_fp16_max, neginf=-_fp16_max)
        else:
            raise ValueError("Unknown normalization method %s", self._normalization)

        attn_output = attn_output if delta_x_offsets[0].shape[0] == 0 else attn_output[delta_x_offsets[0], :]
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
            # norm + ffn + add
            ffn_input = self.norm_ffn(new_outputs)
            new_outputs = self.feed_forward(ffn_input) + new_outputs

        if delta_x_offsets[0].shape[0] > 0:
            new_outputs = cached_outputs.index_copy_(dim=0, index=delta_x_offsets[0], source=new_outputs)

        if return_cache_states and delta_x_offsets[0].shape[0] == 0:
            v = v.contiguous()

        if jagged_enabled:
            new_outputs = torch.ops.fbgemm.jagged_to_padded_dense(
                values=new_outputs,
                offsets=[x_offsets],
                max_lengths=[n],
                padding_value=0.0,
            )

        return new_outputs, (v, q, k, new_outputs), time_bias


@ModelRegistry.register(req_hp=True, opt_subs={"RABModule"})
class FUXI(Transformer):
    """
    HSTU模型用于处理序列数据.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):

        """
        继承父类Transformer的参数
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        ffn_expand = model_cfg[Const.HP].get("ffn_expand")
        ## 判断attention的linear head的数目
        ## 如果attention和位置，时间bias均存在，则linear head的数目为3 + 1 = 4
        if self._normalization == "rel_bias" and self._rel_attn_bias is not None:
            self.linear_number = 4
        ## 如果attention存在，位置，时间bias不存在，则linear head的数目为1 + 1 = 2
        elif self._normalization == "rel_bias" and self._rel_attn_bias is None:
            self.linear_number = 2
        ## 如果attention不存在，位置，时间bias存在，则linear head的数目为2 + 1 = 3            
        elif self._normalization == "att_free_bias" and self._rel_attn_bias is not None:
            self.linear_number = 3
        ## 其他情况下模型报错
        else:
            raise ValueError("error, check the configuration.")

        if self._linear_config == "uvqk":
            self._uvqk = torch.nn.Parameter(
                torch.empty((self._embedding_dim, self.linear_number * self._linear_dim * self._num_heads +
                             self._attention_dim * self._num_heads * 2)).normal_(mean=0, std=0.02), )
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)

        self._o = torch.nn.Linear(
            in_features=(self.linear_number - 1) * self._linear_dim * self._num_heads,
            out_features=self._embedding_dim
        )

        torch.nn.init.xavier_uniform_(self._o.weight)

        self.layer_norm_attn_output = RMSNorm_npu(
            (self.linear_number - 1) * self._linear_dim * self._num_heads,
            eps=self._eps
        )

        self.layer_norm_ffn = RMSNorm_npu(self._embedding_dim, eps=self._eps)

        self.ffn_expand = ffn_expand
        self.feed_forward = FeedForward(
            dim=self._embedding_dim,
            hidden_dim=int(self._embedding_dim * ffn_expand),
            dropout=self._dropout_ratio,
        )

        feat_conf = common_hp.get("feature_conf")
        fuse_ia = feat_conf.get("fuse_ia", False)
        self.token_per_item = 2 if not fuse_ia else 1

    def _norm_ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm_ffn(x)

    def _linear_transform(self, normed_x: torch.Tensor) -> torch.Tensor:
        if self._linear_config == "uvqk":
            batched_mm_output = torch.matmul(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            # u 特征交互, qkv transformer
            u, v, q, k = torch.split(
                batched_mm_output,
                [(self.linear_number - 1) * self._linear_dim * self._num_heads, self._linear_dim * self._num_heads,
                 self._attention_dim * self._num_heads, self._attention_dim * self._num_heads],
                dim=-1,
            )
            return u, v, q, k
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)

    def forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            invalid_attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            cache: TransformerCacheState = (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = torch.Tensor([]),
            _x_offsets_list=None,
            _history_length=None
    ) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor]:
        """
        前向传播方法, 处理输入序列并生成输出序列.

        :param x: 输入序列的特征, 形状为(\sum_i N_i, D).
        :param x_offsets: 输入序列的偏移量, 形状为(B + 1), 表示每个序列的起始位置.
        :param all_timestamps: 可选参数, 时间戳序列, 形状为(B, N).
        :param invalid_attn_mask: 无效的注意力掩码, 形状为(B, N, N), 每个元素为0或1.
        :param past_lengths: 过去序列的长度, 形状为(B,).
        :param num_rerank: 需要推理部分的长度.
        :param layer_num: 当前层的编号.
        :param delta_x_offsets: 可选参数, 形状为((B,), (B,))的偏移量, 对于元组中的第一个元素, 
            每个元素在[0,x_offsets[-1])中. 对于元组中的第2个元素, 每个元素在[0,N)中.
        :param cache: 可选参数, 缓存状态, 用于存储中间结果(v, padded_q, padded_k, output).
        :param return_cache_states: 是否返回缓存状态.
        :param _x_offsets_list: 预计算的x_offsets.tolist()结果, 由TransformerInner传入以避免每层重复NPU→CPU同步.
            为None时在层内调用x_offsets.tolist()作为兜底.
        :param _history_length: 预计算的history_length, 由TransformerInner传入以避免每层RAB模块重复.item() NPU→CPU同步.
        :return: 处理后的输出序列, 形状为(\sum_i N_i, D).
        """
        # n 代表整个需要推理的序列长度
        jagged_enabled = HAS_ATTN_FUSION_OPS and num_rerank == 0 and self._normalization == "rel_bias"
        if jagged_enabled:
            x = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets])[0]
        n: int = invalid_attn_mask.shape[-1]
        if delta_x_offsets[0].shape[0] > 0:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # 维度 [delta_x_offsets[0], :].
            cached_v = torch.zeros_like(x, device=x.device)
            cached_q = torch.zeros_like(x, device=x.device)
            cached_k = torch.zeros_like(x, device=x.device)
            cached_outputs = torch.zeros_like(x, device=x.device)
            if cache[0][0].shape[0] == 0:
                raise ValueError("cache must be provided when delta_x_offsets is not None")
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = cache

        normed_x = self._norm_input(x)

        if self._linear_config == "uvqk":
            u, v, q, k = self._linear_transform(normed_x)
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)

        if delta_x_offsets[0].shape[0] > 0:
            v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        bs: int = x_offsets.shape[0] - 1

        # fuxi-alpha，保留q * k的 attention 计算矩阵
        if self._normalization == "rel_bias":
            if delta_x_offsets[0].shape[0] > 0:
                k, q = handle_padded_qk(bs, cached_k, cached_q, delta_x_offsets, k, n, q)

            if HAS_ATTN_FUSION_OPS:
                attn_mask = invalid_attn_mask.unsqueeze(1)
                if num_rerank == 0:
                    qk_shape = (-1, self._num_heads, self._attention_dim)
                    v_shape = (-1, self._num_heads, self._linear_dim)
                    mask = None
                    mask_type = 0  # 0: tril
                    layout = "jagged"
                    seq_offset = _x_offsets_list if _x_offsets_list is not None else x_offsets.tolist()
                    out_shape = (-1, self._num_heads * self._linear_dim)
                else:
                    qk_shape = (bs, n, self._num_heads, self._attention_dim)
                    v_shape = (bs, n, self._num_heads, self._linear_dim)
                    # 推理时mask做过特殊处理，无法直接使用融合算子内置的mask。 repeat至 [bs, _num_heads, n, n]
                    mask = attn_mask.repeat(1, self._num_heads, 1, 1)
                    mask_type = 3  # 3: custom
                    layout = "normal"
                    seq_offset = None
                    out_shape = (bs, n, self._num_heads * self._linear_dim)

                # mask_type: 0 tril, 1 triu, 2 none, 3 custom. layout: "normal" padding, "jagged" non-padding
                attn_output = torch.ops.mxrec.hstu_dense(
                    q.view(qk_shape), k.view(qk_shape), v.view(v_shape), mask, None, mask_type,
                    n, self.qk_attn_denominator_value, layout, seq_offset
                ).reshape(out_shape)

                if jagged_enabled:
                    attn_output = torch.ops.fbgemm.jagged_to_padded_dense(
                        values=attn_output,
                        offsets=[x_offsets],
                        max_lengths=[n],
                        padding_value=0.0,
                    )
                    x = torch.ops.fbgemm.jagged_to_padded_dense(
                        values=x,
                        offsets=[x_offsets],
                        max_lengths=[n],
                        padding_value=0.0,
                    )
                    v = torch.ops.fbgemm.jagged_to_padded_dense(
                        values=v,
                        offsets=[x_offsets],
                        max_lengths=[n],
                        padding_value=0.0,
                    )
                    u = torch.ops.fbgemm.jagged_to_padded_dense(
                        values=u,
                        offsets=[x_offsets],
                        max_lengths=[n],
                        padding_value=0.0,
                    )
            else:
                # 形如 [B, H, N, N]
                qk_attn = torch.einsum(
                    "bnhd,bmhd->bhnm",
                    q.view(bs, n, self._num_heads, self._attention_dim),
                    k.view(bs, n, self._num_heads, self._attention_dim),
                )

                qk_attn = F.silu(qk_attn) * self.qk_attn_denominator_value
                # FP16 safety: clamp before mask multiply to prevent 0*Inf=NaN
                if qk_attn.dtype == torch.float16:
                    _fp16_max = torch.finfo(qk_attn.dtype).max
                    qk_attn = torch.clamp(qk_attn, min=-_fp16_max, max=_fp16_max)
                attn_mask = invalid_attn_mask.to(qk_attn.device)
                # 形如 [B, 1, N, N]
                attn_mask = attn_mask.unsqueeze(1)
                qk_attn = qk_attn * attn_mask
                # FP16 safety: NaN from 0*Inf in masked positions → 0
                if qk_attn.dtype == torch.float16:
                    qk_attn = torch.nan_to_num(qk_attn, nan=0.0)
                attn_output = torch.einsum(
                    "bhnm,bmhd->bnhd",
                    qk_attn,
                    v.view(bs, n, self._num_heads, self._linear_dim)
                ).reshape(bs, n, self._num_heads * self._linear_dim)
                # FP16 safety: sanitize attention output
                if attn_output.dtype == torch.float16:
                    _fp16_max = torch.finfo(attn_output.dtype).max
                    attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=_fp16_max, neginf=-_fp16_max)

        # fuxi-beta，去掉attention计算矩阵
        elif self._normalization == "att_free_bias":
            attn_mask = invalid_attn_mask.to(q.device)
            # 形如 [B, 1, N, N]
            attn_mask = attn_mask.unsqueeze(1)

        else:
            raise ValueError("Unknown normalization method %s", self._normalization)

        if all_timestamps is not None and self._rel_attn_bias is not None:
            # Relative Attention Bias --> attention bias
            # reshape qk_attn:  batch x num_head x 2n x 2n  --> batch x num_head x n x 2 x n x 2

            if torch.onnx.is_in_onnx_export() or num_rerank > 0:
                # 分档推理
                # rel_attention_mask 形如 [bs, (n-2)//2, (n-2)//2]
                rel_attention_mask, time_bias = self._rel_attn_bias(
                    all_timestamps, past_lengths, num_rerank,
                    layer_num, time_bias, (n - num_rerank - 1) // self.token_per_item,
                    history_length=_history_length)
            else:
                rel_attention_mask, time_bias = self._rel_attn_bias(
                    all_timestamps, past_lengths, num_rerank,
                    layer_num, time_bias, history_length=_history_length)
            # 对于fuxi模型，rab_aggregate_method应该为concat，attention_mask的中间维度应该为2*，判断是否配置错误
            if len(rel_attention_mask.shape) != 4:
                logging.error("the rab_aggregate_method should be configured as concat.")

            # 形如 [bs, 2, (n-1), (n-1)]
            rel_attention_mask = rel_attention_mask.repeat(1, 1, 1, self.token_per_item)  # b, 2, (n-2)/2, n-2
            seq_tokens = n // self.token_per_item - 1
            rel_attention_mask = rel_attention_mask.view(
                bs, 2, seq_tokens, 2, seq_tokens
            ).repeat(1, 1, 1, 1, self.token_per_item)  # b, 2, (n-2)/2, 2, n-2
            rel_attention_mask = rel_attention_mask.transpose(-1, -2).reshape(bs, 2, n - 1, n - 1)
            rel_attention_mask = torch.nn.functional.pad(rel_attention_mask, (1, 0, 1, 0), 'constant', 0.0)
            rel_attention_mask = rel_attention_mask * attn_mask
            rel_attn_output = torch.einsum(
                "bhnm,bmd->bnhd",
                rel_attention_mask,
                v.view(bs, n, self._num_heads * self._linear_dim)
            ).reshape(bs, n, 2 * self._num_heads * self._linear_dim)

        if self._normalization == "rel_bias" and self._rel_attn_bias is not None:
            attn_output = torch.cat([attn_output, rel_attn_output], 2)
        elif self._normalization == "rel_bias" and self._rel_attn_bias is None:
            attn_output = attn_output
        elif self._normalization == "att_free_bias" and self._rel_attn_bias is not None:
            attn_output = rel_attn_output
        else:
            raise ValueError("error, check the configuration.")

        attn_output = attn_output if delta_x_offsets[0].shape[0] == 0 else attn_output[delta_x_offsets[0], :]
        o_input = u * self._norm_attn_output(attn_output)
        # x --> u k q v
        new_outputs = self._o(
            F.dropout(
                o_input,
                p=self._dropout_ratio,
                training=self.training,
            )
        ) + x

        ## fuxi-alpha引入FFN层
        ffn_input = self._norm_ffn(new_outputs)
        ffn_output = self.feed_forward.forward(ffn_input) + new_outputs

        if delta_x_offsets[0].shape[0] > 0:
            ffn_output = cached_outputs.index_copy_(dim=0, index=delta_x_offsets[0], source=ffn_output)

        if return_cache_states and delta_x_offsets[0].shape[0] == 0:
            v = v.contiguous()

        return ffn_output, (v, q, k, ffn_output), time_bias


@ModelRegistry.register(req_hp=True, req_subs={"Transformer"})
class SequentialModule(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        sequential_module_config = model_cfg[Const.HP]
        self.num_blocks = sequential_module_config.get("num_blocks", 8)
        self._transformer = TransformerInner(
            modules=[self.init_sub_model("Transformer") for _ in range(self.num_blocks)]
        )

    def forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            invalid_attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            cache: Optional[List[TransformerCacheState]] = None,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            action_bias: torch.Tensor = None
    ) -> Tuple[torch.Tensor, List[TransformerCacheState]]:
        return self._transformer(
            x=x,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            cache=cache,
            delta_x_offsets=delta_x_offsets,
            return_cache_states=return_cache_states,
            action_bias=action_bias
        )

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass


class TransformerInner(torch.nn.Module):
    """
    TransformerInner类, 用于封装一系列Transformer模块, 如LlaMa, HSTU, Fuxi等, 实现分层序列建模.
    该类负责管理多个Transformer模块, 并提供前向传播接口.
    """

    def __init__(
            self,
            modules: List[Transformer],
    ) -> None:
        super().__init__()
        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=modules)
        # 从第一层获取dynamic padding标志（所有层配置相同）
        self._use_dynamic_padding: bool = modules[0]._use_dynamic_padding if modules else False

    def forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            invalid_attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            cache: Optional[List[TransformerCacheState]] = None,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = torch.tensor([]),
            action_bias: torch.Tensor = None
    ) -> Tuple[torch.Tensor, List[TransformerCacheState]]:
        """
        前向传播方法, 通过多个STU模块处理输入序列.

        :param x: 输入序列的特征, 形状为(\sum_i N_i, D).
        :param x_offsets: 输入序列的偏移量, 形状为(B + 1).
        :param all_timestamps: 时间戳序列, 形状为(B, 1 + N).
        :param invalid_attn_mask: 无效的注意力掩码, 形状为(B, N, N).
        :param past_lengths: 过去序列的长度, 形状为(B,).
        :param num_rerank: 需要推理部分的长度.
        :param cache: 可选参数, 缓存状态列表.
        :param delta_x_offsets: 可选参数, 形状为形状为((B,), (B,))的偏移量.
        :param return_cache_states: 是否返回缓存状态.
        :param time_bias: 时间偏移量.
        :param action_bias: 行为类型注意力偏置, 形状为(B, N, N), 用于PinRec风格的行为条件化.
        :return: 处理后的输出序列.
        """
        cache_states: List[TransformerCacheState] = []

        # 预计算 x_offsets.tolist() 一次，避免每层重复 NPU→CPU 同步
        # (hstu_dense 融合算子需要 Python list 类型的 seq_offset 参数)
        _x_offsets_list = x_offsets.tolist() if HAS_ATTN_FUSION_OPS and num_rerank == 0 else None

        # 预计算 past_lengths.max().item() 一次，仅在dynamic padding时需要
        # 非dynamic padding时RAB使用固定history_length常量，无需NPU→CPU同步
        _hist_len = past_lengths.max().item() if (past_lengths.dim() > 0 and self._use_dynamic_padding) else None

        for i, layer in enumerate(self._attention_layers):
            x, cache_states_i, time_bias = layer(
                x=x,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                past_lengths=past_lengths,
                num_rerank=num_rerank,
                layer_num=i,
                cache=cache[i] if cache is not None else (
                    torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
                delta_x_offsets=delta_x_offsets,
                return_cache_states=return_cache_states,
                time_bias=time_bias,
                action_bias=action_bias,
                _x_offsets_list=_x_offsets_list,
                _history_length=_hist_len
            )
            if return_cache_states:
                cache_states.append(cache_states_i)

        return x, cache_states
