from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional

import torch
from modeling.generic.sequential.attn_mask_modules import AttentionMaskModule
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.embedding_modules import EmbeddingModule
from modeling.generic.sequential.input_features_preprocessors import InputFeaturesPreprocessorModule
from modeling.generic.sequential.loss_modules import LossModule
from modeling.generic.sequential.negative_sampler import NegativesSampler
from modeling.generic.sequential.output_postprocessors import OutputPostprocessorModule
from modeling.generic.sequential.prediction_modules import FeedForwardModule
from modeling.generic.sequential.deep_modules import DLRModule, RankMixer
from modeling.generic.sequential.transformers import SequentialModule, TransformerCacheState
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const


@ModelRegistry.register(
    req_subs={"EmbeddingModule", "InputFeaturesPreprocessorModule",
              "SequentialModule", "AttentionMaskModule",
              "FeedForwardModule", "OutputPostprocessorModule",
              "LossModule", "DLRModule"}, opt_subs={"NegativesSampler"})
class LONGER(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self._verbose = model_cfg[Const.HP].get("verbose", True)
        model_conf = common_hp["model_conf"]
        feat_conf = common_hp["feature_conf"]
        self.seq_id_keys = list(feat_conf.get("seq_feature_columns").keys())
        infer_items_key = feat_conf.get("infer_items_key", "sequence_item_ids")
        self.embedding_module: EmbeddingModule = self.init_sub_model("EmbeddingModule")
        self.input_propcessor_module: InputFeaturesPreprocessorModule = self.init_sub_model(
            "InputFeaturesPreprocessorModule")
        self.deep_model: DLRModule = self.init_sub_model("DLRModule")
        self.sequence_model: SequentialModule = self.init_sub_model("SequentialModule")
        self.negative_sampler: NegativesSampler = None if "NegativesSampler" not in model_cfg[Const.SUB_MODELS] \
            else self.init_sub_model("NegativesSampler")
        if self.negative_sampler is not None:
            self.negative_sampler.load_embedding_module(self.embedding_module)
        seq_feature_columns = common_hp["feature_conf"]["seq_feature_columns"]
        # 在音乐数据中，序列id特征与序列同名，所以用这种方式来取长度
        self.sequence_lengths = torch.tensor([int(v[k]["length"]) for k, v in seq_feature_columns.items()])
        self._seq_lens = [v[k]["length"] for k, v in seq_feature_columns.items()]

        self.infer_items_key = infer_items_key

        self.attention_mask_module: AttentionMaskModule = self.init_sub_model("AttentionMaskModule")
        self.feed_forward_module: FeedForwardModule = self.init_sub_model("FeedForwardModule")
        self.output_processor_module: OutputPostprocessorModule = self.init_sub_model("OutputPostprocessorModule")
        self.loss_module: LossModule = self.init_sub_model("LossModule")

        self.balance_loss_coef = 1e-8
        self.reset_params()

    def reset_params(self):
        for name, params in self.named_parameters():
            if ("sequence_model" in name) or ("embedding_module" in name):
                if self._verbose:
                    logging.info("Skipping init for %s", name)
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    logging.info("Initialize %s as xavier normal: %s params", name, params.data.shape[0])
            except Exception:
                if self._verbose:
                    logging.info("Failed to initialize %s: %s params", name, params.data.shape[0])

    def get_embeddings(self, model_inputs):
        seq_feature_embs = self.embedding_module.get_seq_embeddings(model_inputs)
        user_feature_embs = self.embedding_module.get_user_embeddings(model_inputs)
        item_feature_embs = self.embedding_module.get_item_embeddings(model_inputs)
        return item_feature_embs, user_feature_embs, seq_feature_embs

    def generate_input_seqence(
            self,
            model_inputs,
            past_lengths,
            num_rerank: int
    ) -> torch.Tensor:
        """
        综合序列信息，生成user, item1, action1, item2, action2...形式的输入序列。

        :return user_embeddings: 拼接后的输入给模型的token序列，形如user, item1, action1, item2, action2...
        :return past_embeddings: 原始的商品token序列，形如item1, item2, item3, ...
        :return all_timestamps: 时间戳序列
        """
        past_ids = [model_inputs[k] for k in self.seq_id_keys]
        item_feature_embs, user_feature_embs, seq_feature_embs = self.get_embeddings(model_inputs)
        # 如果是在推理时，将历史序列和候选集序列拼接后返回；如果在训练，则只返回历史序列。
        # deep_outputs形状为 (B, 1, D)或(B, num_rerank, D)
        deep_results = self.deep_model(
            past_ids=past_ids,
            num_rerank=num_rerank,
            model_inputs=model_inputs,
            user_feature_embs_original=user_feature_embs,
            item_feature_embs_original=item_feature_embs,
            seq_feature_embs=seq_feature_embs
        )

        input_seq_embeddings, x_offsets, seq_offsets = self.input_propcessor_module(
            model_inputs=model_inputs,
            past_ids=past_ids,
            num_rerank=num_rerank,
            past_lengths=past_lengths,
            user_feature_embs=user_feature_embs,
            item_feature_embs=item_feature_embs,
            seq_feature_embs=seq_feature_embs,
            deep_outputs=deep_results["deep_outputs"]
        )

        return input_seq_embeddings, deep_results, x_offsets, seq_offsets

    def generate_user_embeddings(
            self,
            past_lengths: torch.Tensor,
            all_timestamps: torch.Tensor,
            seq_embeddings: torch.Tensor,
            deep_outputs: torch.Tensor,
            attn_mask: torch.Tensor,
            x_offsets: torch.Tensor,
            seq_offsets: torch.Tensor,
            cache: Optional[List[TransformerCacheState]] = None,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            num_rerank: int = 1,
    ) -> torch.Tensor:
        """
        综合序列信息，生成用户 embedding.
        [B, N] -> [B, N, D].
        """
        item_embeddings, _ = self.sequence_model(
            x=seq_embeddings,
            x_offsets=x_offsets,
            seq_offsets=seq_offsets,
            all_timestamps=None,
            invalid_attn_mask=attn_mask,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        # LONGER融合方案,取所有的deep_out输出作为输出头
        item_embeddings = item_embeddings[:, : num_rerank, :]

        return self.output_processor_module(item_embeddings + deep_outputs)

    def generate_user_embeddings_prefill(
            self,
            past_lengths: torch.Tensor,
            all_timestamps: torch.Tensor,
            seq_embeddings: torch.Tensor,
            deep_outputs: torch.Tensor,
            attn_mask: torch.Tensor,
            x_offsets: torch.Tensor,
            seq_offsets: torch.Tensor,
            cache: Optional[List[TransformerCacheState]] = None,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            num_rerank: int = 1,
    ) -> torch.Tensor:
        """
        综合序列信息，生成用户 embedding.
        [B, N] -> [B, N, D].
        """
        item_embeddings, cached_states = self.sequence_model(
            x=seq_embeddings,
            x_offsets=x_offsets,
            seq_offsets=seq_offsets,
            all_timestamps=None,
            invalid_attn_mask=attn_mask,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        return cached_states

    def generate_user_embeddings_decode(
            self,
            past_lengths: torch.Tensor,
            all_timestamps: torch.Tensor,
            seq_embeddings: torch.Tensor,
            deep_outputs: torch.Tensor,
            attn_mask: torch.Tensor,
            x_offsets: torch.Tensor,
            seq_offsets: torch.Tensor,
            cache: cache,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            num_rerank: int = 1,
    ) -> torch.Tensor:
        """
        综合序列信息，生成用户 embedding.
        [B, N] -> [B, N, D].
        """
        item_embeddings = self.sequence_model(
            x=seq_embeddings,
            x_offsets=x_offsets,
            seq_offsets=seq_offsets,
            all_timestamps=None,
            invalid_attn_mask=attn_mask,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        item_embeddings = item_embeddings[:, : num_rerank, :]
        return self.output_processor_module(item_embeddings + deep_outputs)

    def forward(
            self,
            model_input: Dict[str, torch.Tensor]
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        生成式推荐大模型前向传播过程

        :param model_input: 传入的字典，里面包括商品的特征id序列，用户的特征和其他序列信息。
        """
        past_ids = [model_input[k] for k in self.seq_id_keys]
        past_lengths = []
        for seq in past_ids:
            past_length = (seq != 0).sum(dim=1).unsqueeze(1)
            past_lengths.append(past_length)
        past_lengths = torch.concat(past_lengths, dim=1)
        model_input.update(past_lengths=past_lengths)
        # num_rerank: target item的数量，取值大于或等于1， 当前约定训练时为1， 推理时大于等于1。
        num_rerank = model_input.get(self.infer_items_key).shape[-1]

        input_seq_embeddings, deep_results, x_offsets, seq_offsets = self.generate_input_seqence(
            model_inputs=model_input,
            past_lengths=past_lengths,
            num_rerank=num_rerank)

        attn_mask = self.attention_mask_module(model_input, self._seq_lens, num_rerank)
        encoded_embeddings = self.generate_user_embeddings(past_lengths=past_lengths,
                                                           all_timestamps=None,
                                                           seq_embeddings=input_seq_embeddings,
                                                           deep_outputs=deep_results["deep_outputs"],
                                                           attn_mask=attn_mask,
                                                           x_offsets=x_offsets,
                                                           seq_offsets=seq_offsets,
                                                           num_rerank=num_rerank)
        results = self.feed_forward_module(encoded_embeddings)
        if torch.onnx.is_in_onnx_export() or not self.training:
            return results
        else:
            loss = self.loss_module(past_embeddings=input_seq_embeddings,
                                    encoded_embeddings=encoded_embeddings,
                                    predictions=results,
                                    model_inputs=model_input,
                                    negative_sampler=self.negative_sampler)
            return loss

    def prefill_forward(
            self,
            model_input: Dict[str, torch.Tensor]
    ) -> torch.Tensor | Dict[str, torch.Tensor]:

        logging.info("this is the prefill_forward in GR_model.py, the main entrance")
        past_ids = [model_input[k] for k in self.seq_id_keys]
        past_lengths = []
        for seq in past_ids:
            past_length = (seq != 0).sum(dim=1).unsqueeze(1)
            past_lengths.append(past_length)
        past_lengths = torch.concat(past_lengths,
                                    dim=1)
        model_input.update(past_lengths=past_lengths)
        # num_rerank: target item的数量，取值大于或等于1， 当前约定训练时为1， 推理时大于等于1。
        num_rerank = model_input.get(self.infer_items_key).shape[-1]

        input_seq_embeddings, deep_results, x_offsets, seq_offsets = self.generate_input_seqence(
            model_inputs=model_input,
            past_lengths=past_lengths,
            num_rerank=num_rerank)
        attn_mask_prefill = self.attention_mask_module(model_input, self._seq_lens, num_rerank)
        prefill_cached_states = self.generate_user_embeddings_prefill(past_lengths=past_lengths,
                                                                      all_timestamps=None,
                                                                      seq_embeddings=input_seq_embeddings,
                                                                      deep_outputs=deep_results["deep_outputs"],
                                                                      attn_mask=attn_mask_prefill,
                                                                      x_offsets=x_offsets,
                                                                      seq_offsets=seq_offsets,
                                                                      num_rerank=num_rerank)

        cached_v = prefill_cached_states.cached_v.detach().unsqueeze(0)
        cached_k = prefill_cached_states.cached_k.detach().unsqueeze(0)
        KVcache = torch.cat((cached_k, cached_v), dim=0).unsqueeze(0)

        return KVcache

    def decode_forward(
            self,
            model_input: Dict[str, torch.Tensor]
    ) -> torch.Tensor | Dict[str, torch.Tensor]:

        logging.info("this is the decode_forward in GR_model.py, the main entrance")
        past_ids = [model_input[k] for k in self.seq_id_keys]
        past_lengths = []
        for seq in past_ids:
            past_length = (seq != 0).sum(dim=1).unsqueeze(1)
            past_lengths.append(past_length)
        past_lengths = torch.concat(past_lengths,
                                    dim=1)
        model_input.update(past_lengths=past_lengths)
        # num_rerank: target item的数量，取值大于或等于1， 当前约定训练时为1， 推理时大于等于1。
        num_rerank = model_input.get(self.infer_items_key).shape[-1]
        KVcache = torch.tensor(model_input['kv_cache'], dtype=torch.float32)

        del model_input['kv_cache']
        input_seq_embeddings, deep_results, x_offsets, seq_offsets = self.generate_input_seqence(
            model_inputs=model_input,
            past_lengths=past_lengths,
            num_rerank=num_rerank)
        model_input["kv_cache"] = KVcache

        attn_mask_decode = self.attention_mask_module(model_input, self._seq_lens, num_rerank)
        encoded_embeddings = self.generate_user_embeddings_decode(past_lengths=past_lengths,
                                                                  all_timestamps=None,
                                                                  seq_embeddings=input_seq_embeddings,
                                                                  deep_outputs=deep_results["deep_outputs"],
                                                                  attn_mask=attn_mask_decode,
                                                                  x_offsets=x_offsets,
                                                                  seq_offsets=seq_offsets,
                                                                  num_rerank=num_rerank,
                                                                  cache=KVcache
                                                                  )
        results = self.feed_forward_module(encoded_embeddings)
        if torch.onnx.is_in_onnx_export() or not self.training:
            return results
        else:
            loss = self.loss_module(past_embeddings=input_seq_embeddings,
                                    encoded_embeddings=encoded_embeddings,
                                    predictions=results,
                                    model_inputs=model_input,
                                    negative_sampler=self.negative_sampler)
            return loss


@ModelRegistry.register(
    req_subs={"EmbeddingModule", "InputFeaturesPreprocessorModule",
              "SequentialModule", "AttentionMaskModule",
              "FeedForwardModule", "OutputPostprocessorModule",
              "LossModule", "DLRModule"}, opt_subs={"NegativesSampler"})
class LongerEp(LONGER):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        model_cfg[Const.SUB_MODELS]["EmbeddingModule"][Const.MODULE_NAME] \
            = "DistributeEmbeddingModuleWithSideInfoLonger"
        model_cfg[Const.SUB_MODELS]["EmbeddingModule"][Const.CLS_NAME] = "DistributeEmbeddingModuleWithSideInfoLonger"
        super().__init__(model_cfg, common_hp, model_cls_dict)

    def get_embeddings(self, model_inputs):
        return self.embedding_module.get_all_embeddings(model_inputs)
