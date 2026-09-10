import abc
import math
from typing import Dict, Tuple, List, Optional
import logging
import torch
import torch_npu
import os
import stat
import pickle
import torch.nn.functional as F
from modeling.generic.sequential.rab_modules import RABModule
from modeling.generic.sequential.utils import handle_padded_qk
from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
from modeling import HAS_ATTN_FUSION_OPS, HAS_JAGGED_OPS
from modeling.generic.utils.jagged_utils import dense_to_jagged, jagged_to_padded_dense
from modeling.generic.utils.hstu_dense_utils import hstu_dense
from modeling.generic.utils.hstu_fuxi_utils import hstu_fuxi

TransformerCacheState = Const.TransformerCacheState


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


@ModelRegistry.register(req_hp=True)
class HSTUCache:
    def __init__(self, n: int = 0):
        self.cached_v = torch.tensor([])
        self.cached_k = torch.tensor([])
        self.n = 0

    def append(self, cache: Tuple[torch.Tensor, torch.Tensor]):
        """
        向缓存中增加新的元素
        """
        k, v = cache

        self.cached_v = torch.cat((self.cached_v, v), dim=0)
        self.cached_k = torch.cat((self.cached_k, k), dim=0)
        self.n += 1

    def select(self, index: int = 0):  # prefill 阶段不需要这个函数
        """
        根据索引取出特定的缓存元素
        """
        if index < 0 or index >= self.n:
            raise IndexError("Index out of range.")

        return self.cached_k[index], self.cached_v[index]


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class FeedForwardV2(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1_d = torch.nn.Linear(dim, dim, bias=False)
        self.w1_u = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(self.w1_u(F.silu(self.w1_d(x)) * self.w3(x))))


class FeedForwardV3(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1_d = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w1_u = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.silu(self.w1_u(self.w1_d(x))) * self.w3(x))


class RMSNorm_npu(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Transformer(BaseModel):
    """
    基础的 Sequential Transduction Unit, STU 用于处理序列数据.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        model_conf = common_hp["model_conf"]

        SeqentialModuleConfig = model_cfg[Const.HP]
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 128)
        self._linear_dim: int = SeqentialModuleConfig.get("dv", 32)
        self._attention_dim: int = SeqentialModuleConfig.get("dqk", 32)
        self._num_heads: int = SeqentialModuleConfig.get("num_heads", 4)
        self._linear_config: str = SeqentialModuleConfig.get("linear_config", "uvqk")
        self._linear_activation: str = SeqentialModuleConfig.get("linear_activation", "silu")
        self._dropout_ratio: float = model_conf.get("linear_dropout_rate", 0.3)
        self._attn_dropout_ratio: float = model_conf.get("attn_dropout_rate", 0.0)
        self._normalization: str = model_conf.get("normalization", "rel_bias")
        self._max_sequence_length: int = model_conf.get("max_sequence_length", 512)
        self._rel_attn_bias: RABModule = self.init_sub_mode("RABModule") if "RABModule" in model_cfg[Const.SUB_MODELS] \
            else None
        self._eps: float = Const.EPS

        self.enable_fusion_ops = SeqentialModuleConfig.get("enable_fusion_ops", False)
        self.enable_jagged_ops = SeqentialModuleConfig.get("enable_jagged_ops", False)

        if not self.enable_fusion_ops and self.enable_jagged_ops:
            raise ValueError("It is impossible to set enable_jagged_ops to " \
                             "True while setting enable_fusion_ops to False")
        elif self.enable_fusion_ops and self.enable_jagged_ops:
            logging.info("Enable fuxi fusion ops according to your configuration.")
        elif self.enable_fusion_ops and not self.enable_jagged_ops:
            logging.info("Enable hstu fusion ops according to your configuration.")
        else:
            logging.info("Enable traditional einsum ops according to your configuration.")

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

        qk_attn_denominator = SeqentialModuleConfig.get("qk_attn_denominator", "emb_dim")
        if qk_attn_denominator == "emb_dim":
            self.qk_attn_denominator_value = 1 / self._embedding_dim
        elif qk_attn_denominator == "sqrt_d":
            self.qk_attn_denominator_value = 1 / math.sqrt(self._embedding_dim)
        elif qk_attn_denominator == "max_seq_len":
            self.qk_attn_denominator_value = 1 / (self._max_sequence_length * 2 + 2)
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
    ) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor, torch.Tensor]:
        pass


@ModelRegistry.register(opt_subs={"RABModule"})
class PassThrough(Transformer):
    """
    HSTU模型用于处理序列数据.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        继承父类Transformer的参数
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self._uvqk = None
        self._o = None
        self.layer_norm_input = None
        self.layer_norm_attn_output = None

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
            time_bias: torch.Tensor = torch.tensor([])
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
        :return: 处理后的输出序列, 形状为(\sum_i N_i, D).
        """

        return x, (None, None, None, x), time_bias


def write_to_file(save_file, mode='w', encoding=None):
    _flags = os.O_WRONLY | os.O_CREAT
    _stats = stat.S_IWUSR | stat.S_IRUSR
    if encoding is not None:
        file_hander = os.fdopen(os.open(save_file, _flags, _stats), mode, encoding=encoding)
    else:
        file_hander = os.fdopen(os.open(save_file, _flags, _stats), mode)
    return file_hander


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

        self._o = torch.nn.Linear(in_features=(self.linear_number - 1) * self._linear_dim * self._num_heads,
                                  out_features=self._embedding_dim)

        torch.nn.init.xavier_uniform_(self._o.weight)

        self.layer_norm_attn_output = RMSNorm_npu((self.linear_number - 1) * self._linear_dim * self._num_heads,
                                                  eps=self._eps)

        self.layer_norm_ffn = RMSNorm_npu(self._embedding_dim, eps=self._eps)

        self.ffn_expand = ffn_expand
        self.feed_forward = FeedForward(
            dim=self._embedding_dim,
            hidden_dim=int(self._embedding_dim * ffn_expand),
            dropout=self._dropout_ratio,
        )

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
            time_bias: torch.Tensor = torch.Tensor([])
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
        :return: 处理后的输出序列, 形状为(\sum_i N_i, D).
        """
        # n 代表整个需要推理的序列长度
        n: int = invalid_attn_mask.shape[-1]
        jagged_enabled = HAS_ATTN_FUSION_OPS and HAS_JAGGED_OPS and \
                         self.training and self.enable_jagged_ops and self.enable_fusion_ops
        fusion_enabled = HAS_ATTN_FUSION_OPS and self.enable_fusion_ops
        cached_v = torch.zeros_like(x, device=x.device)
        cached_q = torch.zeros_like(x, device=x.device)
        cached_k = torch.zeros_like(x, device=x.device)
        cached_outputs = torch.zeros_like(x, device=x.device)
        if delta_x_offsets[0].shape[0] > 0:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # 维度 [delta_x_offsets[0], :].
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

        bs: int = past_lengths.shape[0]

        # fuxi-alpha，保留q * k的 attention 计算矩阵
        if self._normalization == "rel_bias":
            if delta_x_offsets[0].shape[0] > 0:
                k, q = handle_padded_qk(bs, cached_k, cached_q, delta_x_offsets, k, n, q)

            if fusion_enabled:
                invalid_attn_mask = invalid_attn_mask.unsqueeze(1)
                # 训练、评估时mask做过特殊处理，无法直接使用融合算子内置的mask。 repeat至 [bs, _num_heads, n, n]
                mask = invalid_attn_mask.repeat(1, self._num_heads, 1, 1)
                mask_type = 3  # custom
                if jagged_enabled:
                    qk_shape = (-1, self._num_heads, self._attention_dim)
                    v_shape = (-1, self._num_heads, self._linear_dim)
                    layout = "jagged"
                    seq_offset = x_offsets
                    out_shape = (-1, self._num_heads * self._linear_dim)
                else:
                    qk_shape = (bs, n, self._num_heads, self._attention_dim)
                    v_shape = (bs, n, self._num_heads, self._linear_dim)
                    layout = "normal"
                    seq_offset = None
                    out_shape = (bs, n, self._num_heads * self._linear_dim)
                    attn_output = hstu_dense(
                        q.view(qk_shape), k.view(qk_shape), v.view(v_shape), mask, None, mask_type,
                        n, self.qk_attn_denominator_value, layout, seq_offset
                    ).reshape(out_shape)

            else:
                # 形如 [B, H, N, N]
                qk_attn = torch.einsum(
                    "bnhd,bmhd->bhnm",
                    q.view(bs, n, self._num_heads, self._attention_dim),
                    k.view(bs, n, self._num_heads, self._attention_dim),
                )

                qk_attn = F.silu(qk_attn) * self.qk_attn_denominator_value
                invalid_attn_mask = invalid_attn_mask.to(qk_attn.device)
                # 形如 [B, 1, N, N]
                invalid_attn_mask = invalid_attn_mask.unsqueeze(1)
                qk_attn = qk_attn * invalid_attn_mask
                attn_output = torch.einsum(
                    "bhnm,bmhd->bnhd",
                    qk_attn,
                    v.view(bs, n, self._num_heads, self._linear_dim)
                ).reshape(bs, n, self._num_heads * self._linear_dim)

        # fuxi-beta，去掉attention计算矩阵
        elif self._normalization == "att_free_bias":
            invalid_attn_mask = invalid_attn_mask.to(q.device)
            # 形如 [B, 1, N, N]
            invalid_attn_mask = invalid_attn_mask.unsqueeze(1)

        else:
            raise ValueError("Unknown normalization method %s", self._normalization)

        if all_timestamps is not None and self._rel_attn_bias is not None:
            # Relative Attention Bias --> attention bias
            # reshape qk_attn:  batch x num_head x 2n x 2n  --> batch x num_head x n x 2 x n x 2

            if torch.onnx.is_in_onnx_export() or not self.training:
                # 分档推理
                # rel_attention_mask 形如 [bs, (n-2)//2, (n-2)//2]
                rel_attention_mask, time_bias = self._rel_attn_bias(all_timestamps, past_lengths, num_rerank,
                                                                    layer_num,
                                                                    time_bias, (n - num_rerank - 2) // 2)
            else:
                rel_attention_mask, time_bias = self._rel_attn_bias(all_timestamps, past_lengths, num_rerank,
                                                                    layer_num,
                                                                    time_bias)
            # 对于fuxi模型，rab_aggregate_method应该为concat，attention_mask的中间维度应该为2*，判断是否配置错误
            if len(rel_attention_mask.shape) != 4:
                logging.error("the rab_aggregate_method should be configured as concat.")

            # 形如 [bs, 2, (n-2), (n-2)]
            rel_attention_mask = rel_attention_mask.repeat(1, 1, 1, 2)  # b, 2, (n-2)/2, n-2
            rel_attention_mask = rel_attention_mask.view(bs, 2, n // 2 - 1, 2, n // 2 - 1).repeat(1, 1, 1, 1,
                                                                                                  2)
            rel_attention_mask = rel_attention_mask.transpose(-1, -2).reshape(bs, 2, n - 2, n - 2)
            rel_attention_mask = torch.nn.functional.pad(rel_attention_mask, (1, 1, 1, 1), 'constant', 0.0)
            rel_attention_mask = rel_attention_mask * invalid_attn_mask
            rel_attn_output = torch.einsum(
                "bhnm,bmd->bnhd",
                rel_attention_mask,
                v.view(bs, n, self._num_heads * self._linear_dim)
            ).reshape(bs, n, 2 * self._num_heads * self._linear_dim)

        if jagged_enabled:
            attn_output = hstu_fuxi(
                q.view(qk_shape), k.view(qk_shape), v.view(v_shape), None, None, mask,
                mask_type, n, self.qk_attn_denominator_value, layout, seq_offset)
        else:
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

        # fuxi-alpha引入FFN层
        ffn_input = self._norm_ffn(new_outputs)
        ffn_output = self.feed_forward.forward(ffn_input) + new_outputs

        if delta_x_offsets[0].shape[0] > 0:
            ffn_output = cached_outputs.index_copy_(dim=0, index=delta_x_offsets[0], source=ffn_output)

        if return_cache_states and delta_x_offsets[0].shape[0] == 0:
            v = v.contiguous()

        return ffn_output, (v, q, k, ffn_output), time_bias


    def prefill_forward(
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
            time_bias: torch.Tensor = torch.Tensor([])
    ) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor]:

        # n 代表整个需要推理的序列长度
        n: int = invalid_attn_mask.shape[-1]
        cached_v = torch.zeros_like(x,
                                    device=x.device)
        cached_q = torch.zeros_like(x,
                                    device=x.device)
        cached_k = torch.zeros_like(x,
                                    device=x.device)
        cached_outputs = torch.zeros_like(x,
                                          device=x.device)
        if delta_x_offsets[0].shape[0] > 0:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # 维度 [delta_x_offsets[0], :].
            if cache[0][0].shape[0] == 0:
                raise ValueError("cache must be provided when delta_x_offsets is not None")
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = cache

        normed_x = self._norm_input(x)

        if self._linear_config == "uvqk":
            u, v, q, k = self._linear_transform(normed_x)
        else:
            raise ValueError("Unknown linear_config %s",
                             self._linear_config)

        if delta_x_offsets[0].shape[0] > 0:
            v = cached_v.index_copy_(dim=0,
                                     index=delta_x_offsets[0],
                                     source=v)

        bs: int = past_lengths.shape[0]

        # fuxi-alpha，保留q * k的 attention 计算矩阵
        if self._normalization == "rel_bias":
            if delta_x_offsets[0].shape[0] > 0:
                k, q = handle_padded_qk(bs,
                                        cached_k,
                                        cached_q,
                                        delta_x_offsets,
                                        k,
                                        n,
                                        q)
            else:
                # 形如 [B, H, N, N]
                qk_attn = torch.einsum(
                    "bnhd,bmhd->bhnm",
                    q.view(bs, n, self._num_heads,
                           self._attention_dim),
                    k.view(bs, n, self._num_heads,
                           self._attention_dim),
                )

                qk_attn = F.silu(qk_attn) * self.qk_attn_denominator_value
                invalid_attn_mask = invalid_attn_mask.to(qk_attn.device)
                # 形如 [B, 1, N, N]
                invalid_attn_mask = invalid_attn_mask.unsqueeze(1)
                qk_attn = qk_attn * invalid_attn_mask
                attn_output = torch.einsum(
                    "bhnm,bmhd->bnhd",
                    qk_attn,
                    v.view(bs, n,
                           self._num_heads,
                           self._linear_dim)
                ).reshape(bs, n,
                          self._num_heads * self._linear_dim)

        attn_output = attn_output if delta_x_offsets[0].shape[0] == 0 \
            else attn_output[delta_x_offsets[0], :]
        o_input = u * self._norm_attn_output(attn_output)
        # x --> u k q v
        new_outputs = self._o(
            F.dropout(o_input,
                      p=self._dropout_ratio,
                      training=self.training, )
        ) + x

        # fuxi-alpha引入FFN层
        ffn_input = self._norm_ffn(new_outputs)
        ffn_output = self.feed_forward.forward(ffn_input) \
                     + new_outputs

        if delta_x_offsets[0].shape[0] > 0:
            ffn_output = cached_outputs.index_copy_(dim=0,
                                                    index=delta_x_offsets[0],
                                                    source=ffn_output)

        if return_cache_states and delta_x_offsets[0].shape[0] == 0:
            v = v.contiguous()

        return ffn_output, (k, v), time_bias

    def decode_forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            invalid_attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            cache,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = torch.Tensor([])
    ) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor]:

        n: int = invalid_attn_mask.shape[-2]
        m: int = invalid_attn_mask.shape[-1]
        cached_q = torch.zeros_like(x,
                                    device=x.device)
        cached_outputs = torch.zeros_like(x,
                                          device=x.device)
        if delta_x_offsets[0].shape[0] > 0:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # 维度 [delta_x_offsets[0], :].
            if cache[0][0].shape[0] == 0:
                raise ValueError("cache must be provided when delta_x_offsets is not None")
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = \
                cache

        normed_x = self._norm_input(x)

        if self._linear_config == "uvqk":
            u, v, q, k = self._linear_transform(normed_x)
        else:
            raise ValueError("Unknown linear_config %s",
                             self._linear_config)

        if delta_x_offsets[0].shape[0] > 0:
            v = cached_v.index_copy_(dim=0,
                                     index=delta_x_offsets[0],
                                     source=v)

        bs: int = past_lengths.shape[0]
        cached_k = cache[0].unsqueeze(0)
        cached_v = cache[1].unsqueeze(0)
        new_k = torch.cat([k, cached_k], dim=1)
        new_v = torch.cat([v, cached_v], dim=1)

        # fuxi-alpha，保留q * k的 attention 计算矩阵
        if self._normalization == "rel_bias":
            if delta_x_offsets[0].shape[0] > 0:
                k, q = handle_padded_qk(bs,
                                        cached_k,
                                        cached_q,
                                        delta_x_offsets,
                                        k,
                                        n,
                                        q)
            else:
                # 形如 [B, H, N, N]
                qk_attn = torch.einsum(
                    "bnhd,bmhd->bhnm",
                    q.view(bs,
                           n,
                           self._num_heads,
                           self._attention_dim),
                    new_k.view(bs,
                               m,
                               self._num_heads,
                               self._attention_dim),
                )

                qk_attn = F.silu(qk_attn) * \
                          self.qk_attn_denominator_value
                qk_attn = qk_attn * \
                          invalid_attn_mask.unsqueeze(1)

                attn_output = torch.einsum(
                    "bhnm,bmhd->bnhd", qk_attn, new_v.view(bs,
                                                           m,
                                                           self._num_heads,
                                                           self._linear_dim)
                ).reshape(bs,
                          n,
                          self._num_heads * self._linear_dim)

        attn_output = attn_output if delta_x_offsets[0].shape[0] == 0 else \
            attn_output[delta_x_offsets[0], :]
        o_input = u * self._norm_attn_output(attn_output)
        # x --> u k q v
        new_outputs = self._o(
            F.dropout(
                o_input,
                p=self._dropout_ratio,
                training=self.training,
            )) + x

        # fuxi-alpha引入FFN层
        ffn_input = self._norm_ffn(new_outputs)
        ffn_output = self.feed_forward.forward(ffn_input) + new_outputs

        if delta_x_offsets[0].shape[0] > 0:
            ffn_output = cached_outputs.index_copy_(dim=0,
                                                    index=delta_x_offsets[0],
                                                    source=ffn_output)

        if return_cache_states and delta_x_offsets[0].shape[0] == 0:
            v = v.contiguous()

        return ffn_output, time_bias


@ModelRegistry.register(req_hp=True, req_subs={"Transformer"})
class SequentialModule(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        SeqentialModuleConfig = model_cfg[Const.HP]
        self.enable_jagged_ops = model_cfg[Const.SUB_MODELS]["Transformer"][Const.HP].get("enable_jagged_ops",
                                                                                          False)
        self.enable_fusion_ops = model_cfg[Const.SUB_MODELS]["Transformer"][Const.HP].get("enable_fusion_ops",
                                                                                          False)
        self.num_blocks = SeqentialModuleConfig.get("num_blocks", 8)
        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=[self.init_sub_model("Transformer") for _ in range(self.num_blocks)])

    def forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            seq_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            invalid_attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            cache: Optional[List[TransformerCacheState]] = None,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = torch.tensor([])
    ) -> Tuple[torch.Tensor, List[TransformerCacheState]]:

        cache_states: List[TransformerCacheState] = []

        jagged_enabled = HAS_ATTN_FUSION_OPS and HAS_JAGGED_OPS and \
                         self.training and self.enable_jagged_ops and self.enable_fusion_ops

        if jagged_enabled:
            seq_offsets = list(seq_offsets)
            jagged_length = int(seq_offsets[-1])
            dense_length: int = invalid_attn_mask.shape[-1]
            x = dense_to_jagged(x, x_offsets, dense_length, jagged_length)

        for i, layer in enumerate(self._attention_layers):
            x, cache_states_i, time_bias = layer(
                x=x,
                x_offsets=seq_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                past_lengths=past_lengths,
                num_rerank=num_rerank,
                layer_num=i,
                cache=cache[i] if cache is not None else (
                    torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
                delta_x_offsets=delta_x_offsets,
                return_cache_states=return_cache_states,
                time_bias=time_bias
            )
            if return_cache_states:
                cache_states.append(cache_states_i)

        if jagged_enabled:
            x = jagged_to_padded_dense(
                x,
                x_offsets,
                dense_length,
                jagged_length
            )
        return x, cache_states

    def prefill_forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            seq_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            invalid_attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            cache: Optional[List[TransformerCacheState]] = None,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = torch.tensor([])
    ) -> Tuple[torch.Tensor, List[TransformerCacheState]]:

        cache_states: HSTUCache = HSTUCache(0)

        for i, layer in enumerate(self._attention_layers):
            x, cache_states_i, time_bias = layer(
                x=x,
                x_offsets=seq_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                past_lengths=past_lengths,
                num_rerank=num_rerank,
                layer_num=i,
                cache=cache[i] if cache is not None else (
                    torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
                delta_x_offsets=delta_x_offsets,
                return_cache_states=return_cache_states,
                time_bias=time_bias
            )
            cache_states.append(cache_states_i)

        return x, cache_states

    def decode_forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            seq_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            invalid_attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            cache: HSTUCache,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = torch.tensor([])
    ) -> Tuple[torch.Tensor, HSTUCache]:

        for i, layer in enumerate(self._attention_layers):
            x, time_bias = layer(
                x=x,
                x_offsets=seq_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                past_lengths=past_lengths,
                num_rerank=num_rerank,
                layer_num=i,
                cache=torch.squeeze(cache[:, :, i, :, :]),
                delta_x_offsets=delta_x_offsets,
                return_cache_states=return_cache_states,
                time_bias=time_bias
            )

        return x

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

