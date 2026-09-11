import abc
import logging
import math
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from modeling.generic.initialization import truncated_normal
from modeling.generic.sequential.action_conditioning import ActionConditioningModule
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.utils.constants import Const, FeatConst
from modeling.model_registry import ModelRegistry
from utils.common_utils import weird_division


class RoPEWithPadding(torch.nn.Module):
    """
    对带有padding的序列应用RoPE位置编码
    仅对非padding部分（非零向量）应用编码，padding部分保持为0
    """

    def __init__(self, dim, max_seq_len=200):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 计算频率（采用LLaMA的RoPE计算方式）
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 预先计算位置编码
        self.register_buffer('pe', self._compute_pe(max_seq_len))

    def _compute_pe(self, max_seq_len):
        """预先计算所有可能位置的RoPE编码"""
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        freqs = torch.matmul(position, self.inv_freq.unsqueeze(0))

        # 计算sin和cos
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return emb.unsqueeze(0)  # 形状: [1, max_seq_len, dim]

    def forward(self, x):
        """
        对输入序列应用RoPE编码，忽略padding部分

        参数:
            x: 输入张量，形状为[B, N, D]，其中包含padding（尾部连续0）

        返回:
            编码后的张量，形状与输入相同，padding部分保持为0
        """
        batch_size, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"输入维度 {dim} 与初始化维度 {self.dim} 不匹配")

        # 找到每个序列的有效长度（非padding部分）
        # 判断每个位置是否为padding（全零向量）
        is_padding = (x == 0).all(dim=-1)  # 形状: [B, N]

        # 计算每个序列的有效长度（第一个padding出现的位置）
        # 对于全非padding的序列，有效长度为seq_len
        valid_lengths = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        for i in range(batch_size):
            # 找到第一个padding的索引
            pad_indices = torch.where(is_padding[i])[0]
            if len(pad_indices) > 0:
                valid_lengths[i] = pad_indices[0]
            else:
                valid_lengths[i] = seq_len

        # 初始化输出张量
        out = torch.zeros_like(x)

        # 对每个样本单独处理
        for i in range(batch_size):
            vl = valid_lengths[i]
            if vl == 0:
                continue  # 整个序列都是padding，直接跳过

            # 提取有效部分
            x_valid = x[i, :vl, :]

            # 获取对应长度的位置编码
            rope = self.pe[:, :vl, :]

            # 应用RoPE编码
            x_rot = x_valid[..., :self.dim]

            # 按奇偶维度拆分
            x1 = x_rot[..., ::2]  # 偶数索引
            x2 = x_rot[..., 1::2]  # 奇数索引

            # 旋转操作
            cos = rope[..., 1::2]
            sin = rope[..., ::2]

            x1_rot = x1 * cos - x2 * sin
            x2_rot = x1 * sin + x2 * cos

            # 合并旋转后的结果
            x_rot = torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)

            # 将编码结果放入输出张量的对应位置
            out[i, :vl, :] = x_rot

        return out


class InputFeaturesPreprocessorModule(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp.get("model_conf", {})
        train_conf = common_hp.get("train_conf", {})
        self._bf16_mode = (train_conf.get("use_amp", False) and
                           model_conf.get("amp_dtype", "fp16") == "bf16")

    @abc.abstractmethod
    def process_rerank_embs(
            self,
            rerank_embs: torch.Tensor,
            past_lengths: torch.Tensor,  # B, 1
    ):
        pass

    @abc.abstractmethod
    def forward(
            self,
            past_lengths: torch.Tensor,
            past_ids: torch.Tensor,
            user_feature_embs: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pass


@ModelRegistry.register()
class UserItemRatingInputFeaturePreprocessor(InputFeaturesPreprocessorModule):
    """
    用户-物品-评分输入特征预处理模块, 用于处理用户、物品和评分的特征。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        model_conf = common_hp.get("model_conf")
        feat_conf = common_hp.get("feature_conf")
        max_sequence_length = model_conf.get("max_sequence_length", 512)
        gr_output_length = model_conf.get("gr_output_length", 0)
        self.max_sequence_len = max_sequence_length + gr_output_length
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)
        num_ratings = model_conf.get("num_ratings", 5)
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            self.max_sequence_len * 2 + 1, self._embedding_dim,
        )
        self._dropout_rate: float = model_cfg[Const.HP].get("embedding_dropout_rate", 0.3)
        self._emb_dropout = torch.nn.Dropout(p=self._dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings + 1, self._embedding_dim, padding_idx=0
        )
        self.num_ratings = num_ratings
        self._infer_ratings_key = feat_conf.get("infer_ratings_key", "ratings")
        self.reset_state()

    def debug_str(self) -> str:
        return f"combir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data, mean=0.0, std=math.sqrt(weird_division(1.0, self._embedding_dim)),
        )
        truncated_normal(
            self._rating_emb.weight.data, mean=0.0, std=math.sqrt(weird_division(1.0, self._embedding_dim)),
        )

    def get_preprocessed_masks(
            self,
            past_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        生成预处理后的掩码。

        :param past_ids: 历史ID张量。
        :return: 预处理后的掩码张量, 形状为(B, N * 2)。
        """
        B, N = past_ids.size()
        return (past_ids != 0).unsqueeze(2).expand(-1, -1, 2).reshape(B, N * 2)

    def forward(
            self,
            past_lengths: torch.Tensor,
            past_ids: torch.Tensor,
            user_feature_embs: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
            num_rerank: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播方法, 用于处理用户、物品和评分的特征。

        :param past_lengths: 历史长度张量, 形状为(B,), 其中B是批次大小, 表示每个序列的长度。
        :param past_ids: 历史ID张量, 形状为(B, N), 其中N是序列中的最大项数, 表示每个序列中的物品ID。
        :param user_feature_embs: 用户特征embedding张量, 形状为(B, D), D是用户特征embedding的维度。
        :param past_embeddings: 历史embedding张量, 形状为(B, N, D), 包含序列中每个物品的embedding表示。
        :param past_payloads: 历史信息字典, 包含序列中每个项的额外信息, 如评分和时间戳。
        :return: 新的序列长度, 形状为(B,), 是原始有效序列长度的两倍+1。
                 预处理后的用户特征embedding, 形状为(B, 1+N*2+1, D), 将用户特征、物品特征和评分特征结合起来最后补0。
                 有效掩码张量, 形状为(B, N*2), 用于指示哪些位置是有效的, 即非零ID的位置。
        """
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        # 提取评分 embedding
        rating_emb = self._rating_emb(past_payloads[self._infer_ratings_key])

        # 拼接历史物品 embedding 与评分 embedding, (i1,i2,i3,...), (a1,a2,a3,...)->(i1,a1,i2,a2,i3,a3,...)
        user_embeddings = torch.cat(
            [
                past_embeddings,
                rating_emb
            ], dim=2,
        ) * (self._embedding_dim ** 0.5)
        user_embeddings = user_embeddings.view(B, N * 2, D)
        user_embeddings = (
                user_embeddings
                + self._pos_emb(torch.arange(N * 2, device=past_ids.device).unsqueeze(0).repeat(B, 1))
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        # 生成有效掩码并应用
        mask_dtype = torch.bfloat16 if self._bf16_mode else torch.float32
        valid_mask = self.get_preprocessed_masks(
            past_ids
        ).unsqueeze(2).to(mask_dtype)
        user_embeddings *= valid_mask
        user_embeddings = torch.cat((user_feature_embs.unsqueeze(1), user_embeddings), dim=1)
        # # 为了适用于底层加速，需要使user_embeddings序列长度为偶数

        return 1 + past_lengths * 2, user_embeddings, valid_mask

    def process_rerank_embs(
            self,
            rerank_embs: torch.Tensor,  # B, NUM_CANDIDATE, D
            past_lengths: torch.Tensor,  # B, 1
    ):
        B, N, D = rerank_embs.shape
        position_embs = self._pos_emb(past_lengths - 1).unsqueeze(1).repeat(1, N, 1)  # B x NUM_CANDIDATE x D
        rerank_embs = rerank_embs * (self._embedding_dim ** 0.5) + position_embs
        rerank_embs = torch.nn.functional.pad(rerank_embs, (0, 0, 0, 1, 0, 0), 'constant', 0.0)
        return rerank_embs

    def get_num_ratings(self):
        return self.num_ratings


@ModelRegistry.register()
class UserItemInputFeaturePreprocessor(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        model_conf = common_hp.get("model_conf")
        feat_conf = common_hp.get("feature_conf")
        max_sequence_length = model_conf.get("max_sequence_length", 512)
        gr_output_length = model_conf.get("gr_output_length", 0)
        self.max_sequence_len = max_sequence_length + gr_output_length
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)
        num_ratings = model_conf.get("num_ratings", 5)

        self._dropout_rate: float = model_cfg[Const.HP].get("embedding_dropout_rate", 0.3)
        self._emb_dropout = torch.nn.Dropout(p=self._dropout_rate)

        self.num_ratings = num_ratings
        self._hist_infer_ratings_key = feat_conf.get("history_ratings_column", "history_action_type")
        self._cand_infer_ratings_key = feat_conf.get("candidate_ratings_column", "candidate_action_type")

        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            self.max_sequence_len + 1, self._embedding_dim,
        )

        model_conf = common_hp.get("model_conf")
        if model_conf.get("use_action_emb", True):
            self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
                num_ratings + 2, self._embedding_dim
            )
        else:
            self._rating_emb = None

        self.mask_emb_id = self.num_ratings + 1
        train_conf = common_hp["train_conf"]
        self._phase = train_conf.get("phase", "pretrain")

        # sid_fusion_pos: 决定 SID 融合方式（model 或 input）
        self.sid_fusion_pos = model_cfg.get(Const.HP, {}).get("sid_fusion_pos", "model")
        self._raw_sid_dim = 0
        if self.sid_fusion_pos == "input":
            model_hp = model_cfg.get(Const.HP, {})
            use_sid_config = model_conf.get("use_sid", False)
            if use_sid_config:
                sid_D = feat_conf.get("sid_D", 64)
                num_code_layers = len(feat_conf.get("sid_K", [256, 256, 256]))
                self._raw_sid_dim = sid_D * num_code_layers
                logging.info("UserItemInputFeaturePreprocessor: sid_fusion_pos=input,"
                             " raw_sid_dim=%s, embedding dim adjusted", self._raw_sid_dim)

        self._use_pos_emb = train_conf.get("_use_pos_emb", True)
        if self._use_pos_emb:
            self._pos_aligned_side = train_conf.get("_pos_aligned_side", "right")
            logging.info("_use_pos_emb")

        self._add_pos_emb = train_conf.get("add_pos_emb", False)
        if self._add_pos_emb:
            logging.info("add pos emb")

        self._add_rope_emb = train_conf.get("add_rope_emb", False)
        if self._add_rope_emb:
            self.rope = RoPEWithPadding(dim=self._embedding_dim, max_seq_len=max_sequence_length)
            logging.info("add rope emb")

        self._add_fixed_pos_emb = train_conf.get("add_fixed_pos_emb", True)
        if self._add_fixed_pos_emb:
            logging.info("add fixed pos emb")

        self._add_fixed_time_aware_pos_emb = train_conf.get("add_fixed_time_aware_pos_emb", True)
        if self._add_fixed_time_aware_pos_emb:
            logging.info("add fixed time aware pos emb")

        # Action conditioning module (PinRec-inspired)
        self.use_action_conditioning = model_conf.get("use_action_conditioning", False)
        if self.use_action_conditioning:
            self.action_conditioning = ActionConditioningModule(
                action_emb_dim=self._embedding_dim,
                token_emb_dim=self._embedding_dim,
                num_action_types=num_ratings,
                use_film=model_conf.get("use_film", True),
                use_gated_fusion=model_conf.get("use_gated_fusion", True),
                use_attention_biasing=model_conf.get("use_attention_biasing", True),
                film_hidden_dim=model_conf.get("film_hidden_dim", 128),
                gate_hidden_dim=model_conf.get("gate_hidden_dim", 64),
                attention_bias_scale=model_conf.get("attention_bias_scale", 0.1)
            )
            logging.info("Action conditioning enabled: FiLM=%s, GatedFusion=%s, AttentionBiasing=%s" % (
                model_conf.get("use_film", True),
                model_conf.get("use_gated_fusion", True),
                model_conf.get("use_attention_biasing", True)
            ))

        model_conf = common_hp.get("model_conf", {})
        self._bf16_mode = (train_conf.get("use_amp", False) and
                           model_conf.get("amp_dtype", "fp16") == "bf16")
        self.reset_state()

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data, mean=0.0, std=math.sqrt(weird_division(1.0, self._embedding_dim)),
        )
        if self._rating_emb:
            truncated_normal(
                self._rating_emb.weight.data, mean=0.0, std=math.sqrt(weird_division(1.0, self._embedding_dim)),
            )

    def get_preprocessed_masks(
            self,
            past_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        生成预处理后的掩码。

        :param past_ids: 历史ID张量。
        :return: 预处理后的掩码张量, 形状为(B, N)。
        """
        B, N = past_ids.size()
        return (past_ids != 0).reshape(B, N)

    def forward(
            self,
            history_embeddings: torch.Tensor,
            candidate_embeddings: torch.Tensor,
            history_lengths: torch.Tensor,
            history_ids: torch.Tensor,
            candidate_ids: torch.Tensor,
            user_feature_embs: torch.Tensor,
            history_ratings: Union[torch.Tensor, int],
            candidate_ratings: torch.Tensor,
            candi_times=None,
            hist_times=None,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        1. 除以特征维度； 2. 拼接history+candidate; 3. dropout, 4. 有效item的掩码

        :param history_embeddings: 历史embedding张量, 形状为(B, N, D), 包含历史序列中每个物品的embedding表示。
        :param candidate_embeddings: 候选embedding张量, 形状为(B, M, D), 包含candidate序列中每个物品的embedding表示。
        :param history_lengths: 历史长度张量, 形状为(B,), 其中B是批次大小, 表示每个历史序列的有效长度（即不是padding的长度）。
        :param history_ids: 历史ID张量, 形状为(B, N), 其中N是序列中的最大项数, 表示每个序列中的物品ID。
        :param candidate_ids: 后续ID张量, 形状为(B, M), 其中N是序列中的最大项数, 表示每个序列中的物品ID。
        :param user_feature_embs: 用户特征embedding张量, 形状为(B, D), D是用户特征embedding的维度。
        timestamps: hist, cand
        :return: 新的序列长度, 形状为(B,), 是原始有效历史序列长度+1。
                 预处理后的用户特征embedding, 形状为(B, 1+N+M, D), 将用户特征、物品特征和评分特征结合起来最后补0。
                 有效掩码张量, 形状为(B, N), 用于指示哪些位置是有效的, 即非零ID的位置。
        """
        # sid_fusion_pos="input" 在此 Preprocessor (HSTU 路径) 不支持
        if self.sid_fusion_pos == "input" and self._raw_sid_dim > 0:
            raise RuntimeError(
                "UserItemInputFeaturePreprocessor (HSTU 路径) 不支持 sid_fusion_pos='input'，"
                "请使用 sid_fusion_pos='model'。")

        if self._phase == "pretrain":
            # 1. 除以特征维度
            if history_embeddings.size(-1) != self._embedding_dim:
                raise ValueError(f"历史嵌入维度 {history_embeddings.size(-1)} 与期望维度 {self._embedding_dim} 不匹配")
            if candidate_embeddings.size(-1) != self._embedding_dim:
                raise ValueError(
                    f"候选嵌入维度 {candidate_embeddings.size(-1)} 与期望维度 {self._embedding_dim} 不匹配")

            history_embeddings = history_embeddings * (self._embedding_dim ** 0.5)  # (B, N, D)
            candidate_embeddings = candidate_embeddings * (self._embedding_dim ** 0.5)  # (B, M ,D)

            mask_dtype = torch.bfloat16 if self._bf16_mode else torch.float32
            hist_valid_mask = self.get_preprocessed_masks(
                history_ids
            ).unsqueeze(2).to(mask_dtype)
            candi_valid_mask = self.get_preprocessed_masks(
                candidate_ids
            ).unsqueeze(2).to(mask_dtype)
            history_embeddings *= hist_valid_mask
            candidate_embeddings *= candi_valid_mask

            return history_lengths, history_embeddings, candidate_embeddings
        else:
            # 1. 除以特征维度
            if history_embeddings.size(-1) != self._embedding_dim:
                raise ValueError(f"历史嵌入维度 {history_embeddings.size(-1)} 与期望维度 {self._embedding_dim} 不匹配")
            if candidate_embeddings.size(-1) != self._embedding_dim:
                raise ValueError(
                    f"候选嵌入维度 {candidate_embeddings.size(-1)} 与期望维度 {self._embedding_dim} 不匹配")

            history_embeddings = history_embeddings * (self._embedding_dim ** 0.5)  # (B, N, D)
            if self._add_rope_emb:
                history_embeddings = self.rope(history_embeddings)
            candidate_embeddings = candidate_embeddings * (self._embedding_dim ** 0.5)  # (B, M ,D)
            # 2. 拼接history+candidate
            user_embeddings = torch.cat([history_embeddings, candidate_embeddings], dim=1)  # (B, N+M, D)
            if self._rating_emb:
                if isinstance(history_ratings, int):
                    with torch.no_grad():
                        range_row = torch.arange(history_embeddings.shape[1],
                                                 device=history_embeddings.device).unsqueeze(0)
                        lengths_col = history_lengths.unsqueeze(1)
                        history_ratings = (range_row < lengths_col).to(torch.int64) * history_ratings
                hist_rating_emb = self._rating_emb(history_ratings)

                fake_candidate_ratings = torch.ones_like(candidate_ratings) * self.mask_emb_id
                cand_rating_emb = self._rating_emb(fake_candidate_ratings)
                user_rating_embeddings = torch.cat([hist_rating_emb, cand_rating_emb], dim=1)
                try:
                    user_embeddings = user_embeddings + user_rating_embeddings
                except ValueError as e:
                    logging.info('user_embeddings shape: %s', user_embeddings.shape)
                    logging.info('user_rating_embeddings shape: %s', user_rating_embeddings.shape)
                    raise e

                # Apply action conditioning (FiLM and Gated Fusion)
                if self.use_action_conditioning:
                    # Combine history and candidate ratings for action types
                    combined_ratings = torch.cat([history_ratings, candidate_ratings], dim=1)
                    user_embeddings = self.action_conditioning(
                        x=user_embeddings,
                        action_embeddings=user_rating_embeddings,
                        action_types=combined_ratings
                    )
            B, N, D = user_embeddings.size()
            if self._use_pos_emb:
                if self._pos_aligned_side == 'left':
                    pos_ids = torch.concat(
                        [torch.arange(history_embeddings.shape[-2]),
                         torch.ones(candidate_embeddings.shape[-2], dtype=torch.int64) * (
                                 history_embeddings.shape[-2] + 1)],
                        dim=-1
                    ).unsqueeze(0).to(history_ids.device)  # [N + M] -> [1, N + M]
                elif self._pos_aligned_side == 'right':
                    history_pos_ids = torch.arange(history_embeddings.shape[-2], dtype=torch.int64,
                                                   device=history_embeddings.device).unsqueeze(0)  # [L] -> [1, L]
                    history_pos_ids = torch.maximum(
                        history_lengths.unsqueeze(-1) - history_pos_ids,
                        torch.tensor(0, dtype=history_pos_ids.dtype, device=history_pos_ids.device))  # [B, L]
                    candidate_pos_ids = torch.ones(*candidate_embeddings.shape[:-1], dtype=torch.int64,
                                                   device=history_embeddings.device) * (
                                                history_embeddings.shape[-2] + 1)
                    pos_ids = torch.concat([history_pos_ids, candidate_pos_ids], dim=-1)  # [B, L + C]
                    pos_ids = pos_ids.to(torch.int64)
                else:
                    raise ValueError(f'pos_aligned_side must be chosen in ["left", "right"]!!!')
                user_embeddings = user_embeddings + self._pos_emb(pos_ids)
            elif self._add_pos_emb:
                user_embeddings = (
                        user_embeddings
                        + self._pos_emb(torch.arange(N, device=history_ids.device).unsqueeze(0).repeat(B, 1))
                )
            elif self._add_fixed_pos_emb:
                valid_hist_mask = (history_ids != 0)
                valid_candidate_mask = (candidate_ids != 0)

                base_hist_pos_ids = torch.arange(1, history_ids.shape[1] + 1, device=history_ids.device).expand_as(
                    history_ids)
                hist_pos_ids = torch.where(valid_hist_mask, base_hist_pos_ids, 0)

                hist_lengths = valid_hist_mask.sum(dim=-1, keepdim=True)
                candidate_target_pos = hist_lengths + 1
                candidate_pos_ids = torch.where(valid_candidate_mask, candidate_target_pos, 0)

                full_pos_ids = torch.cat([hist_pos_ids, candidate_pos_ids], dim=1)
                user_embeddings = (
                        user_embeddings
                        + self._pos_emb(full_pos_ids)
                )
            elif self._add_fixed_time_aware_pos_emb:
                valid_hist_mask = (history_ids != 0)
                base_hist_pos_ids = torch.arange(1, history_ids.shape[1] + 1, device=history_ids.device)
                # 通过mask将padding位置置为0
                hist_pos_ids = base_hist_pos_ids.unsqueeze(0) * valid_hist_mask

                valid_candidate_mask = (candidate_ids != 0)
                hist_lengths = valid_hist_mask.sum(dim=-1, keepdim=True)

                if candi_times is not None and hist_times is not None:
                    candi_times_exp = candi_times.unsqueeze(-1)
                    hist_times_exp = hist_times.unsqueeze(1)
                    valid_hist_mask_exp = valid_hist_mask.unsqueeze(1)

                    comparison_mask = (hist_times_exp <= candi_times_exp).int() * valid_hist_mask_exp.int()

                    num_hist_before = comparison_mask.int().sum(dim=-1)

                    candidate_target_pos = num_hist_before + 1
                else:
                    candidate_target_pos = hist_lengths + 1

                candidate_pos_ids = candidate_target_pos * valid_candidate_mask

                full_pos_ids = torch.cat([hist_pos_ids, candidate_pos_ids], dim=1)
                user_embeddings = (
                        user_embeddings
                        + self._pos_emb(full_pos_ids)
                )
            # 3. dropout
            user_embeddings = self._emb_dropout(user_embeddings)

            # 4. 生成有效掩码并应用; 最终只看非0的item_id
            mask_dtype = torch.bfloat16 if self._bf16_mode else torch.float32
            past_ids = torch.cat([history_ids, candidate_ids], dim=1)  # (B, N+M)
            valid_mask = self.get_preprocessed_masks(
                past_ids
            ).unsqueeze(2).to(mask_dtype)
            user_embeddings = torch.cat((user_feature_embs, user_embeddings), dim=1)  # (B, 1+N, D)

            return 1 + history_lengths, user_embeddings, valid_mask  # past_lengths加了1个 user 特征 token


@ModelRegistry.register()
class MixFormerFeaturePreprocessor(BaseModel):
    """
    为 MixFormer 模型定制的输入预处理模块。

    将 MTGR 现有的 flat embedding 输入 (user_feature_embs, history_embeddings,
    candidate_embeddings) 处理为 MixFormerModule 期望的 dict 格式:

        emb_dict = {
            'user':     (B, 1, n_u, D),   # 用户特征
            'history':  (B, N, D),        # 历史序列
            'candidate':(B, M, n_c, D),   # 候选物品
        }

    其中 n_u/n_c 来自 feature_conf, D 为 item_embedding_dim。
    user_feature_embs 和 candidate_embeddings 通过小 MLP 投影到 n_u*D / n_c*D，
    然后 reshape 出 head 维度。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        model_conf = common_hp.get("model_conf", {})
        feat_conf = common_hp.get("feature_conf", {})
        dataloader_conf = common_hp.get("data_loader_conf", {})
        model_hp = model_cfg.get(Const.HP, {})

        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)
        self._history_length = dataloader_conf.get("history_length", 150)
        self._num_rerank = dataloader_conf.get("num_rerank", 400)

        # n_u / n_c 来自 feature_conf
        self.n_u = feat_conf.get("n_u", 1)
        self.n_c = feat_conf.get("n_c", 3)

        # 计算各 group 的输入维度以决定 MLP 映射
        feats_dim = self._get_feat_dims(feat_conf)
        feature_groups = feat_conf.get("feature_groups", {})
        user_group = feature_groups.get(FeatConst.USER_PFX, {})
        cand_group = feature_groups.get(FeatConst.CAND_PFX, {})

        user_input_dim = self._calculate_group_dim(user_group.get("features", []), feats_dim)
        cand_input_dim = self._calculate_group_dim(cand_group.get("features", []), feats_dim)

        # sid_fusion_pos="input": 让 cand_mlp 输入维度包含 raw_sid_dim
        self._raw_sid_dim = 0
        sid_fusion_pos = model_hp.get("sid_fusion_pos", "model")
        use_sid_config = model_hp.get("use_sid", False)
        if sid_fusion_pos == "input" and use_sid_config:
            sid_D = model_hp.get("sid_D", feat_conf.get("sid_D", 64))
            num_code_layers = len(model_hp.get("sid_Ks", feat_conf.get("sid_K", [256, 256, 256])))
            sid_agg_type = model_hp.get("sid_agg_type", feat_conf.get("sid_agg_type", "concat"))
            self._raw_sid_dim = sid_D * num_code_layers if sid_agg_type == "concat" else sid_D
            cand_input_dim += self._raw_sid_dim
            logging.info("MixFormerFeaturePreprocessor: sid_fusion_pos=input,"
                         " sid_agg_type=%s, cand_input_dim += %s -> %s",
                         sid_agg_type, self._raw_sid_dim, cand_input_dim)

        # user MLP: user_input_dim -> n_u * D
        if user_input_dim != self.n_u * self._embedding_dim:
            self._user_mlp = nn.Linear(user_input_dim, self.n_u * self._embedding_dim)
            logging.info("MixFormerFeaturePreprocessor: user_mlp %s -> %s",
                         user_input_dim, self.n_u * self._embedding_dim)
        else:
            self._user_mlp = nn.Identity()

        # candidate MLP: cand_input_dim -> n_c * D
        if cand_input_dim != self.n_c * self._embedding_dim:
            self._candidate_mlp = nn.Linear(cand_input_dim, self.n_c * self._embedding_dim)
            logging.info("MixFormerFeaturePreprocessor: cand_mlp %s -> %s",
                         cand_input_dim, self.n_c * self._embedding_dim)
        else:
            self._candidate_mlp = nn.Identity()

        # 可选: action embedding
        self._use_action_emb: bool = model_conf.get("use_action_emb", False)
        num_ratings = feat_conf.get("num_ratings", 5)
        if self._use_action_emb:
            self._rating_emb = nn.Embedding(num_ratings + 2, self._embedding_dim)
            self.mask_emb_id = num_ratings + 1
        else:
            self._rating_emb = None

        # 可选: position embedding (从 train_conf 读取, 与原始 InputFeaturesPreprocessorModule 一致)
        train_conf = common_hp.get("train_conf", {})
        self._use_pos_emb: bool = train_conf.get("_use_pos_emb", True)
        if self._use_pos_emb:
            self._pos_aligned_side = train_conf.get("_pos_aligned_side", "right")
            max_seq_len = self._history_length + self._num_rerank + 2
            self._pos_emb = nn.Embedding(max_seq_len + 2, self._embedding_dim)

        # dropout
        self._dropout_rate: float = model_hp.get("embedding_dropout_rate", 0.2)
        self._emb_dropout = nn.Dropout(p=self._dropout_rate)

        self._hist_infer_ratings_key = feat_conf.get("history_ratings_column", "history_action_type")
        self._cand_infer_ratings_key = feat_conf.get("candidate_ratings_column", "candidate_action_type")

        self.reset_state()

    def reset_state(self) -> None:
        if self._use_pos_emb:
            truncated_normal(
                self._pos_emb.weight.data, mean=0.0,
                std=math.sqrt(weird_division(1.0, self._embedding_dim)),
            )
        if self._rating_emb is not None and hasattr(self._rating_emb, "weight"):
            truncated_normal(
                self._rating_emb.weight.data, mean=0.0,
                std=math.sqrt(weird_division(1.0, self._embedding_dim)),
            )

    @staticmethod
    def _get_feat_dims(feature_conf: Dict) -> Dict[str, int]:
        """获取每个 feature 的维度信息。"""
        candidate_feature_columns = feature_conf.get("candidate_item_feature_columns", {})
        history_feature_columns = feature_conf.get("history_item_feature_columns", {})
        user_feature_columns = feature_conf.get("user_feature_columns", {})

        all_columns = {}
        all_columns.update(candidate_feature_columns)
        all_columns.update(history_feature_columns)
        all_columns.update(user_feature_columns)

        feats_dim = {}
        for feature_name, feature_info in all_columns.items():
            feature_dtype = feature_info.get("dtype", "int")
            if feature_dtype == "con":
                feats_dim[feature_name] = 1
            elif feature_dtype in ("int", "context"):
                feats_dim[feature_name] = feature_info.get("dim", 64)
            elif feature_dtype == "multi":
                shared_feat_name = feature_info.get("shared_feat_name", "")
                if shared_feat_name and shared_feat_name not in feats_dim:
                    shared_info = all_columns.get(shared_feat_name, {})
                    feats_dim[shared_feat_name] = shared_info.get("dim", 64)
                feats_dim[feature_name] = feats_dim.get(shared_feat_name, 64)
        return feats_dim

    @staticmethod
    def _calculate_group_dim(feature_names, feats_dim: Dict[str, int]) -> int:
        input_dim = 0
        for feat_name in feature_names:
            input_dim += feats_dim.get(feat_name, 64)
        return input_dim

    def get_preprocessed_masks(self, history_ids, candidate_ids):
        valid_hist = (history_ids != 0)
        valid_cand = (candidate_ids != 0)
        return valid_hist, valid_cand

    def forward(
            self,
            history_embeddings: torch.Tensor,
            candidate_embeddings: torch.Tensor,
            history_lengths: torch.Tensor,
            history_ids: torch.Tensor,
            candidate_ids: torch.Tensor,
            user_feature_embs: torch.Tensor,
            history_ratings: torch.Tensor,
            candidate_ratings: torch.Tensor,
            **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            history_embeddings: (B, N, D)
            candidate_embeddings: (B, M, D)
            history_lengths: (B,)
            history_ids: (B, N)
            candidate_ids: (B, M)
            user_feature_embs: (B, D_user)
            history_ratings: (B, N)
            candidate_ratings: (B, M)

        Returns:
            new_sequence_lengths: (B,)
            emb_dict: dict with keys 'user', 'history', 'candidate'
            valid_mask: dict with keys 'history', 'candidate'
        """
        B, N, _ = history_embeddings.shape
        _, M, _ = candidate_embeddings.shape

        valid_hist_mask, valid_cand_mask = self.get_preprocessed_masks(history_ids, candidate_ids)

        # ========== 1. History processing ==========
        # * sqrt(D) 是 Transformer 初始化约定的一部分。在 RMSNorm 下该缩放实际被归一化消除，
        # 但保留此操作以与参考代码对齐，避免训练动态的微小差异。
        # 使用固定的 self._embedding_dim (item_embedding_dim) 而非从 tensor shape 动态取值，
        # 防止 history/candidate embedding 维度不一致时出 bug。

        if self._use_action_emb and self._rating_emb is not None:
            if isinstance(history_ratings, int):
                with torch.no_grad():
                    range_row = torch.arange(N, device=history_embeddings.device).unsqueeze(0)
                    lengths_col = history_lengths.unsqueeze(1)
                    history_ratings = (range_row < lengths_col).to(torch.int64) * history_ratings
            if history_ratings.dim() == 1:
                history_ratings = history_ratings.unsqueeze(1).expand(-1, N)
            elif history_ratings.dim() == 2 and history_ratings.shape[1] == 1:
                history_ratings = history_ratings.expand(-1, N)
            hist_rating_emb = self._rating_emb(history_ratings)
            history_embeddings = history_embeddings + hist_rating_emb

        if self._use_pos_emb and hasattr(self, "_pos_emb"):
            if self._pos_aligned_side == "right":
                pos_ids = torch.arange(N, dtype=torch.long, device=history_embeddings.device).unsqueeze(0)
                pos_ids = (history_lengths.long().unsqueeze(-1) - pos_ids).clamp_min(0)
            else:
                pos_ids = torch.arange(N, dtype=torch.long, device=history_embeddings.device).unsqueeze(0)
            history_embeddings = history_embeddings + self._pos_emb(pos_ids) * (self._embedding_dim ** 0.5)

        history_embeddings = self._emb_dropout(history_embeddings)

        # # ========== 2. Candidate processing ==========

        candidate_embeddings = self._emb_dropout(candidate_embeddings)

        # Project candidate: (B, M, D) -> (B, M, n_c * D) -> (B, M, n_c, D)
        candidate_embeddings = self._candidate_mlp(candidate_embeddings)
        candidate_reshaped = candidate_embeddings.view(B, M, self.n_c, self._embedding_dim)
        if self._use_action_emb and self._rating_emb is not None:
            fake_cand_ratings = candidate_ratings.new_full(candidate_ratings.shape, self.mask_emb_id, dtype=torch.long)
            cand_rating_emb = self._rating_emb(fake_cand_ratings)
            candidate_reshaped = candidate_reshaped + cand_rating_emb.unsqueeze(2)

        if self._use_pos_emb and hasattr(self, "_pos_emb"):
            cand_pos_ids = candidate_ids.new_full(
                (1, M), N, dtype=torch.long
            ).expand(B, -1)
            candidate_reshaped = candidate_reshaped + \
                                 self._pos_emb(cand_pos_ids).unsqueeze(2) * (self._embedding_dim ** 0.5)

        # ========== 3. User processing ==========
        user_expanded = self._user_mlp(user_feature_embs)
        user_reshaped = user_expanded.view(B, 1, self.n_u, self._embedding_dim)

        emb_dict = {
            "user": user_reshaped,
            "history": history_embeddings,
            "candidate": candidate_reshaped,
        }

        valid_mask = {
            "history": valid_hist_mask,
            "candidate": valid_cand_mask,
        }

        # 为了对齐原MTGR模型将用户信息单独作为一个token，在mixformer中无意义
        new_sequence_lengths = self.n_u + history_lengths

        return new_sequence_lengths, emb_dict, valid_mask

    def process_rerank_embs(
            self,
            rerank_embs: torch.Tensor,
            past_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Inference-time rerank embedding processing (unused for MixFormer)."""
        return rerank_embs

    def debug_str(self) -> str:
        return f"MixFormerFeaturePreprocessor_nu{self.n_u}_nc{self.n_c}_d{self._dropout_rate}"
