"""
MixFormerModel: 在 MTGR 框架中以配置驱动的 MixFormer 精排模型。

通过继承 MultiSeqMTGRModel, 覆写 process_single_act_seq 和
generate_user_embeddings, 使用 MixFormerFeaturePreprocessor 产出 dict,
然后由 MixFormerSequentialModule 消费。
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch.autograd.profiler import record_function

from modeling.generic.sequential.mixformer_sequential_module import MixFormerSequentialModule
from modeling.generic.sequential.multi_seq_mtgr_model import MultiSeqMTGRModel
from modeling.generic.sequential.transformers import TransformerCacheState
from modeling.generic.utils.constants import FeatConst
from modeling.model_registry import ModelRegistry


@ModelRegistry.register(
    req_subs={"EmbeddingModule", "InputFeaturesPreprocessorModule", "SequentialModule", "AttentionMaskModule",
              "FeedForwardModule", "OutputPostprocessorModule", "LossModule", "SIM"},
    opt_subs={"NegativesSampler"},
)
class MixFormerModel(MultiSeqMTGRModel):
    """
    MixFormer 集成模型的入口类。

    继承自 MultiSeqMTGRModel 以复用其 embedding 查找、DLRM、
    FeedForward、Loss 等下游模块, 仅覆写序列建模部分：
      - InputFeaturesPreprocessorModule -> MixFormerFeaturePreprocessor
      - SequentialModule -> MixFormerSequentialModule
      - process_single_act_seq / generate_user_embeddings -> dict-based 流程
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        # 先调用 MultiSeqMTGRModel.__init__, 但会在其中 init 标准的 sub_modules。
        # 我们在 init 后立即用自己的 MixFormerSequentialModule 替换
        # self.sequence_model。
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        # 将 sequence_model 替换为 MixFormerSequentialModule wrapper
        # 注意: sequence_model 已经被 MultiSeqMTGRModel.__init__ 创建,
        # 我们重新创建 MixFormerSequentialModule 实例
        self.sequence_model: MixFormerSequentialModule = self.init_sub_model("SequentialModule")

        model_conf = common_hp.get("model_conf", {})
        feat_conf = common_hp.get("feature_conf", {})
        self.n_c = feat_conf.get("n_c", 3)

    def process_single_act_seq(
            self,
            action: str,
            num_rerank: int,
            candidate_embeddings: torch.Tensor,
            user_feature_embs: torch.Tensor,
            model_input: dict,
            precomputed_max_seq_len: int = None,
    ) -> torch.Tensor:
        """
        处理单个行为序列。

        与父类不同点：
        1. 使用 MixFormerFeaturePreprocessor (self.input_propcessor_module) 产出 emb_dict。
        2. 将 emb_dict 注入 self.sequence_model。
        3. generate_user_embeddings 使用 dict-based 流程。
        """
        single_act_seq_input = {}
        act_ts_key = self.hist_ts_key
        # query_mixer需要历史序列+候选集的timesteps生成mask
        act_ts = torch.concat(
            [model_input.get(self.hist_ts_key), model_input.get(self.cand_ts_key)], dim=1)

        batch_size = candidate_embeddings.shape[0]

        for feat_name in self.feature_groups[FeatConst.HIST_PFX][action]["features"]:
            single_act_seq_input[feat_name] = model_input[feat_name]
        for feat_name in self.feature_groups[FeatConst.CAND_PFX]["features"]:
            single_act_seq_input[feat_name] = model_input[feat_name]
        single_act_seq_input[self.cand_ratings_key] = model_input[self.cand_ratings_key]
        single_act_seq_input["action_dates_key"] = model_input[self.hist_date_key]
        single_act_seq_input["cand_dates_key"] = model_input[self.cand_date_key]
        single_act_seq_input["history_timestamps"] = model_input[act_ts_key]
        single_act_seq_input[act_ts_key] = model_input[act_ts_key]

        _batch_max_seq_len = (
            precomputed_max_seq_len if precomputed_max_seq_len is not None else
            (model_input.get("history_lengths", self.history_lengths).cpu().max().item() + num_rerank)
            if (self.use_dynamic_padding and model_input.get("history_lengths") is not None)
            else self._max_sequence_length
        )
        attn_mask = self.attention_mask_module(single_act_seq_input, _batch_max_seq_len, num_rerank)

        past_embeddings = self.embedding_module.get_ui_embeddings(
            group_name=FeatConst.HIST_PFX,
            input_features=single_act_seq_input,
            action_type=action,
        )

        if self.use_sid and self.sid_fusion_pos == "model":
            hist_sid_embs = self.get_hist_sid_emb(model_input)
            if self.use_sid_gate:
                g1 = self.hist_sid_gate(torch.cat([past_embeddings, hist_sid_embs], dim=-1))
                past_embeddings = past_embeddings + g1 * hist_sid_embs
            else:
                past_embeddings = past_embeddings + hist_sid_embs

        if self.use_enhanced_interest_embeddings:
            candidate_emb_for_srn = candidate_embeddings
            max_seq_len = past_embeddings.shape[1]
            batch_size = past_embeddings.shape[0]
            if self.use_dynamic_padding:
                _hist_lengths = model_input.get("history_lengths", self.history_lengths)
                if _hist_lengths.dim() == 0:
                    _hist_lengths = _hist_lengths.unsqueeze(0).expand(batch_size)
                sequence_mask = torch.arange(max_seq_len, device=past_embeddings.device)[None,
                                :] < _hist_lengths.unsqueeze(1)
            else:
                sequence_mask = torch.arange(max_seq_len, device=past_embeddings.device)[None, :] < self.history_lengths
                sequence_mask = sequence_mask.expand(batch_size, -1)
            try:
                enhanced_past_embeddings = self.srn_module(
                    [candidate_emb_for_srn, past_embeddings],
                    mask=sequence_mask,
                )
                past_embeddings = past_embeddings + enhanced_past_embeddings
            except Exception as e:
                logging.info("SRN module error: %s", e)
                raise e

        # ---------- MixFormer 特殊处理 ----------
        # 1. 调用 MixFormerFeaturePreprocessor 产出 emb_dict
        with record_function("## InputPreprocessor ##"):
            if self._profile_time:
                _t0 = time.time()
            past_lengths_after_input_processor, emb_dict, _ = self.input_propcessor_module(
                history_embeddings=past_embeddings,
                candidate_embeddings=candidate_embeddings,
                history_lengths=model_input.get(
                    "history_lengths", self.history_lengths
                ) if self.use_dynamic_padding else self.history_lengths.expand(batch_size),
                history_ids=model_input.get(self.hist_items_key),
                candidate_ids=model_input.get(self.cand_items_key),
                user_feature_embs=user_feature_embs,
                history_ratings=self.action_mapping.get(action),
                candidate_ratings=model_input.get(self.cand_ratings_key),
                hist_times=act_ts,
            )
            if self._profile_time:
                logging.info("[PROFILE] MixFormer InputPreprocessor cost: %.3f s", time.time() - _t0)

        # 2. 注入 emb_dict 到 sequence_model
        self.sequence_model.set_emb_dict(emb_dict)

        # 3. 调用 generate_user_embeddings (已覆写)
        with record_function("## MixFormerSequential ##"):
            if self._profile_time:
                _t0 = time.time()
            encoded_embeddings = self.generate_user_embeddings(
                past_lengths=past_lengths_after_input_processor,
                all_timestamps=act_ts,
                seq_embeddings=None,  # 不使用 flat tensor
                attn_mask=attn_mask,
                num_rerank=num_rerank,
            )
            if self._profile_time:
                logging.info("[PROFILE] MixFormer Sequential cost: %.3f s", time.time() - _t0)

        return encoded_embeddings

    def generate_user_embeddings(
            self,
            past_lengths: torch.Tensor,
            all_timestamps: torch.Tensor,
            seq_embeddings: torch.Tensor,
            attn_mask: torch.Tensor,
            cache: Optional[List[TransformerCacheState]] = None,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            num_rerank: int = 0,
            x_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        覆写父类 generate_user_embeddings。

        调用 self.sequence_model (MixFormerSequentialModule) 得到 dict 结果,
        从中提取 candidate 部分用于下游 DLRM / FeedForward 处理。
        """
        if x_offsets is None:
            x_offsets = torch.cat((
                torch.zeros(1, dtype=past_lengths.dtype, device=past_lengths.device),
                torch.cumsum(past_lengths, dim=0),
            ), dim=0)

        result_dict, _ = self.sequence_model(
            x=seq_embeddings,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=attn_mask,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )

        # result_dict['candidate'] 是 (B, M, n_c * D)
        # 注意: 不能调用 self.output_processor_module (L2NormEmbeddingPostprocessor),
        # 因为它会截断到 item_embedding_dim (256), 而 MixFormer 输出 n_c*D (768) 维.
        # MixFormer 的输出直接进入 FeedForwardModule.
        item_embeddings = result_dict["candidate"]

        if self.concat_user_embeddings:
            user_part = result_dict["user"]  # (B, 1, n_u, D)
            user_part = user_part.view(*user_part.shape[:2], -1).expand(-1, item_embeddings.shape[1], -1)
            item_embeddings = torch.cat([user_part, item_embeddings], dim=-1)

        return item_embeddings
