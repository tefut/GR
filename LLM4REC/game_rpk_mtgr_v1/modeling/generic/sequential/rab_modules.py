from typing import Dict, Tuple

import torch

from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.utils.almightygosu_utils import index_select
from modeling.generic.utils.constants import Const
from modeling.model_registry import ModelRegistry


@ModelRegistry.register(req_subs={"RelativeTimeEncoder", "RelativePositionEncoder"})  # 未改动
class RABModule(BaseModel):
    """
    RAB模块

    :param rab_aggregate_method: rab相对时间和相对位置编码聚合的方式，可选'sum'或'concat'或None
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp.get("model_conf")
        self._max_seq_len: int = model_conf.get("max_sequence_length", 256)
        self._num_layers: int = model_conf.get("num_blocks", 8)
        self.aggregate = model_cfg[Const.HP].get("rab_aggregate_method", "sum")
        # RAB模块必须在submodel里指定一个时间编码器和一个位置编码器
        if "RelativeTimeEncoder" in model_cfg[Const.SUB_MODELS]:
            self.rel_t_encoder = self.init_sub_model("RelativeTimeEncoder")
        else:
            raise ValueError("A RelativeTimeEncoder should be assigned in sub_models")
        if "RelativePositionEncoder" in model_cfg[Const.SUB_MODELS]:
            self.rel_p_encoder = self.init_sub_model("RelativePositionEncoder")
        else:
            raise ValueError("A RelativePositionEncoder should be assigned in sub_models")

        feat_conf = common_hp.get("feature_conf")
        fuse_ia = feat_conf.get("fuse_ia", False)
        self.token_per_item = 2 if not fuse_ia else 1

    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length=None,
            history_length=None

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = all_timestamps.shape[0]
        rel_ts_bias = self.rel_t_encoder(
            all_timestamps=all_timestamps,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            layer_num=layer_num,
            time_bias=time_bias,
            length=length,
            history_length=history_length
        )
        rel_pos_bias = self.rel_p_encoder(
            all_timestamps=all_timestamps,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            layer_num=layer_num,
            time_bias=time_bias,
            length=length,
            history_length=history_length
        )

        if self.aggregate == 'sum':
            return rel_pos_bias + rel_ts_bias, time_bias
        elif self.aggregate == 'concat':
            if num_rerank == 0:
                rel_pos_bias = rel_pos_bias.repeat(bs, 1, 1)
            return torch.stack([rel_pos_bias, rel_ts_bias], dim=1), time_bias

        # None, 直接输出
        else:
            raise ValueError(f"{self.aggregate} is not Implemented.")


class RelativeTimeEncoder(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp.get("model_conf", {})
        train_conf = common_hp.get("train_conf", {})
        self._bf16_mode = (train_conf.get("use_amp", False) and
                           model_conf.get("amp_dtype", "fp16") == "bf16")

    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length=None,
            history_length=None
    ):
        pass


@ModelRegistry.register()
class BucketRelativeTimeEncoder(RelativeTimeEncoder):
    """
    HSTU论文提出的基于分桶 + 索引的相对时间编码器

    :param num_buckets: 分桶的数量
    :param bucketization_divisor: 分桶时的除数
    :param use_fbgemm: 是否使用fbgemm
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._max_seq_len: int = model_conf.get("max_sequence_length", 256)
        self._num_layers: int = model_conf.get("num_blocks", 8)
        self._num_buckets = model_cfg[Const.HP].get("num_buckets", 48)
        self._bucketization_divisor = model_cfg[Const.HP].get("bucketization_divisor", 0.301)
        self._ts_w = torch.nn.Parameter(
            torch.empty((self._num_layers, self._num_buckets + 1)).normal_(mean=0, std=0.02),
        ).contiguous()
        self._use_fbgemm = model_cfg[Const.HP].get("use_fbgemm", True)

    # 相对时间编码，HSTU论文原始的编码方式，分桶操作
    def bucketization_fn(self, x: torch.Tensor):
        return (torch.log(torch.abs(x.detach()).clamp(min=1)) / self._bucketization_divisor).long()

    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length: int = None,
            history_length=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播方法, 计算相对注意力偏置.

        :param all_timestamps: 时间戳张量, 形状为[B, N].
        :param num_rerank: 需要推理部分的长度.
        :param layer_num: 层编号.
        :param time_bias: 时间偏置, 可选.
        :param history_length: 预计算的历史长度(避免per-layer .item()调用).
        :return: 相对注意力偏置张量, 形状为[B, N, N].
        """

        bs = all_timestamps.shape[0]

        # 只有导出推理模型时会非None
        if length is None:
            length = all_timestamps.shape[1]

        if not layer_num < self._num_layers:
            raise ValueError("layer_num %s is out of bounds for num_layers %s", layer_num, self._num_layers)

        # 相对时间编码，HSTU论文原始的编码方式，基于分桶 + 索引
        # 形如 [bs, n, n]
        # FP16 max ≈ 65504, timestamp diffs can exceed this → overflow risk
        # BF16 max ≈ 3.4e38, safe for timestamp differences
        ts_diff = all_timestamps.unsqueeze(2) - all_timestamps.unsqueeze(1)
        ts_float_dtype = torch.bfloat16 if self._bf16_mode else torch.float32
        bucketed_timestamps = torch.clamp(
            self.bucketization_fn(ts_diff.to(ts_float_dtype)),
            min=0,
            max=self._num_buckets,
        ).detach()

        rel_ts_bias = torch.tensor([])
        rel_ts_bias = index_select(x=self._ts_w[layer_num, :], index=bucketed_timestamps.view(-1)).view(bs, length,
                                                                                                        length)

        return rel_ts_bias


@ModelRegistry.register()
class PowRelativeTimeEncoder(RelativeTimeEncoder):
    """
    fuxi-beta论文提出的相对时间编码，基于指数函数拟合
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._max_seq_len: int = model_conf.get("max_sequence_length", 256)
        self._num_layers: int = model_conf.get("num_blocks", 8)
        self._a = torch.nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2))
        self._d = torch.nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(0.2, 1.5))
        self._e = torch.nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(0.4, 0.8))

    def f(self, x):
        x = torch.relu(x) + 1
        return self._a / (1 + self._d * torch.pow(x, self._e))

    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length=None,
            history_length=None
    ):
        # 只有导出推理模型时会非None
        if length is None:
            length = self._max_seq_len
        if not layer_num < self._num_layers:
            raise ValueError("layer_num %s is out of bounds for num_layers %s", layer_num, self._num_layers)
        ext_timestamps = (all_timestamps.unsqueeze(2) - all_timestamps.unsqueeze(1)).to(
            torch.bfloat16 if self._bf16_mode else torch.float32)
        rel_ts_bias = self.f(ext_timestamps)
        return rel_ts_bias


@ModelRegistry.register()
class PowRelativeTimeEncoderv2(RelativeTimeEncoder):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._max_seq_len: int = model_conf.get("max_sequence_length", 256)
        self._num_layers: int = model_conf.get("num_blocks", 8)
        self._a = torch.nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.6, 0.6))
        self._e = torch.nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(0.4, 0.6))
        self._use_next_timestamp = model_cfg[Const.HP].get("use_next_timestamp", False)

    def f(self, x):
        self._e.data = torch.clamp(self._e.data, min=0.0001)
        x = torch.abs(x)
        return self._a * torch.pow(0.8, torch.pow(x, self._e))

    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length=None,
            history_length=None
    ):
        # 只有导出推理模型时会非None
        if length is None:
            length = self._max_seq_len
        N = self._max_seq_len
        if self._use_next_timestamp:
            all_timestamps = torch.cat([all_timestamps, all_timestamps[:, N - 1: N]], dim=1)
            timestamp_diffs = all_timestamps[:, 1:].unsqueeze(2) - all_timestamps[:, :-1].unsqueeze(1)
        else:
            timestamp_diffs = all_timestamps.unsqueeze(2) - all_timestamps.unsqueeze(1)
        rel_ts_bias = self.f(timestamp_diffs.to(torch.bfloat16 if self._bf16_mode else torch.float32))
        return rel_ts_bias


class RelativePositionEncoder(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp.get("model_conf", {})
        train_conf = common_hp.get("train_conf", {})
        self._bf16_mode = (train_conf.get("use_amp", False) and
                           model_conf.get("amp_dtype", "fp16") == "bf16")

    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length=None,
            history_length=None
    ):
        pass


@ModelRegistry.register()
class DefaultRelativePositionEncoder(RelativePositionEncoder):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._max_seq_len: int = model_conf.get("max_sequence_length", 256)
        self._num_layers: int = model_conf.get("num_blocks", 8)
        self.history_length: int = common_hp['data_loader_conf'].get('history_length', 400)
        self.use_dynamic_padding = common_hp['data_loader_conf'].get('use_dynamic_padding', False)
        self._pos_w = torch.nn.Parameter(
            torch.empty((2 * (self._max_seq_len + 1) - 1, self._num_layers)).normal_(mean=0, std=0.02),
        )

    def get_time_position_ids(self, timestamps: torch.Tensor, history_length: int) -> torch.Tensor:
        return batched_get_time_position_ids(timestamps, history_length)

    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length=None,
            history_length=None
    ):
        bs = all_timestamps.shape[0]
        if length is None:
            length = all_timestamps.shape[1]

        r = (2 * self._max_seq_len - 1) // 2
        pos_w_slice = self._pos_w[:2 * self._max_seq_len - 1, layer_num]

        if self.use_dynamic_padding:
            # 使用预计算的history_length避免per-layer .item() NPU→CPU同步
            _hist_len = (
                history_length if history_length is not None
                else (past_lengths.max().item() if past_lengths.dim() > 0 else self.history_length))
        else:
            _hist_len = self.history_length
        position_ids = self.get_time_position_ids(all_timestamps, _hist_len)
        rel_index = position_ids.unsqueeze(2) - position_ids.unsqueeze(1)
        rel_index = (rel_index + r).clamp(0, 2 * self._max_seq_len - 2)

        rel_pos_bias = torch.tensor([])
        rel_pos_bias = index_select(x=self._pos_w[layer_num, :], index=rel_index.view(-1)).view(bs, length, length)

        return rel_pos_bias


def batched_get_time_position_ids(timestamps: torch.Tensor, history_length: int) -> torch.Tensor:
    """
    向量化替代原per-sample Python循环的get_time_position_ids。

    timestamps: (B, N+M) — 左填充的历史时间戳 + 可选的候选时间戳
    history_length: int, 前N列为history部分
    返回: position_ids (B, N+M)

    训练时M=0（无候选），直接返回arange(N)跳过所有计算。
    推理时M>0，用比较计数替代searchsorted：count(valid_history < candidate) ≡ searchsorted_left。
    """
    B, L = timestamps.shape
    N = history_length
    M = L - N

    history_pos_ids = torch.arange(N, device=timestamps.device).unsqueeze(0).expand(B, -1)

    if M == 0:
        return history_pos_ids

    history_ts = timestamps[:, :N]  # [B, N]
    candidate_ts = timestamps[:, N:]  # [B, M]

    # 有效时间戳掩码：左填充用0填充，有效时间戳>0（Unix时间戳恒正）
    valid_mask = (history_ts != 0)  # [B, N]

    # 批量searchsorted-left：对每个候选，统计严格小于它的有效历史时间戳数
    compare = (candidate_ts.unsqueeze(2) > history_ts.unsqueeze(1)) & valid_mask.unsqueeze(1)
    candidate_pos_ids = compare.int().sum(dim=2)

    position_ids = torch.cat([history_pos_ids, candidate_pos_ids], dim=1)
    return position_ids


@ModelRegistry.register()
class AlibiRelativePositionEncoder(RelativePositionEncoder):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        sequential_module_config = model_cfg[Const.HP]
        self._max_seq_len: int = model_conf.get("max_sequence_length", 256)
        self._num_layers: int = model_conf.get("num_blocks", 8)
        self._num_heads: int = sequential_module_config.get("num_heads", 4)
        self.history_length: int = common_hp['data_loader_conf'].get('history_length', 400)
        self.use_dynamic_padding = common_hp['data_loader_conf'].get('use_dynamic_padding', False)

        # 初始化每个 head 的 slope
        slopes = self._get_alibi_slope(self._num_heads)
        self.register_buffer("_slopes", slopes.view(1, self._num_heads, 1, 1))  # shape: (1, H, 1, 1)

    def get_time_position_ids(self, timestamps: torch.Tensor, history_length: int) -> torch.Tensor:
        return batched_get_time_position_ids(timestamps, history_length)

    def _get_alibi_slope(self, num_heads: int) -> torch.Tensor:
        """
        生成slope，越后面的head的slope越小
        """
        base = 2 ** 8
        x = base ** (1.0 / num_heads)
        slopes = [1.0 / (x ** (i + 1)) for i in range(num_heads)]
        return torch.tensor(slopes, dtype=torch.float32)

    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length: int = None,
            history_length=None
    ) -> torch.Tensor:
        if self.use_dynamic_padding:
            # 使用预计算的history_length避免per-layer .item() NPU→CPU同步
            _hist_len = (
                history_length if history_length is not None
                else (past_lengths.max().item() if past_lengths.dim() > 0 else self.history_length))
        else:
            _hist_len = self.history_length
        position_ids = self.get_time_position_ids(all_timestamps, _hist_len)
        B, L = position_ids.shape if length is None else (position_ids.shape[0], length)

        # BF16有与FP32相同的指数范围，rel_dist不会溢出，保持BF16避免dtype promotion开销
        # FP16下仍需FP32保护
        rel_dist = position_ids.unsqueeze(2) - position_ids.unsqueeze(1)
        rel_dist = rel_dist.abs()
        if not self._bf16_mode:
            rel_dist = rel_dist.float()
        rel_dist = rel_dist.unsqueeze(1)

        # 乘以每个head的slope
        rel_pos_bias_multihead = -rel_dist * self._slopes  # (B, H, L, L)

        # 平均聚合所有head
        rel_pos_bias = rel_pos_bias_multihead.mean(dim=1)  # (B, L, L)

        return rel_pos_bias
