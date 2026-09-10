from __future__ import annotations

import torch
import torch.nn as nn

from recall.model.embedding_modules import LocalEmbeddingModule
from recall.model.input_features_preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor, \
    HybridInputFeaturesPreprocessor, BehaviorInputFeaturesPreprocessor
from recall.model.output_postprocessors import L2NormEmbeddingPostprocessor, LayerNormEmbeddingPostprocessor
from recall.model.negative_sampler import InBatchNegativesSampler, LocalNegativesSampler, \
    HybridLocalNegativesSampler
from recall.model.autoregressive_losses import SampledSoftmaxLoss, BCELoss
from modeling.generic.sequential.transformers import FUXI

from recall.model.similarity_module.dot_product import DotProductSimilarity
from recall.model.similarity_module.mol import create_mol_interaction_module
from recall.model.utils import get_current_embeddings


class SeqModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self._bulid_model(model_args)

    def _bulid_model(self, model_args):
        self.use_behavior = model_args.get("use_behavior", True)
        self.side_flag = model_args.get("side_flag", False)

        embedding_module_config = model_args.get("embedding_module_config")
        embedding_module_type = embedding_module_config.pop("type")
        if embedding_module_type == "local":
            self.embedding_module = LocalEmbeddingModule(
                **embedding_module_config
            )
        else:
            self.embedding_module = None
            raise ValueError(f"Unrecognized embedding module {embedding_module_type}.")

        input_preproc_module_config = model_args.get("input_preproc_module_config")
        input_preproc_module_type = input_preproc_module_config.pop("type")
        if input_preproc_module_type == "hybrid_input":
            self.input_preproc_module = HybridInputFeaturesPreprocessor(
                **input_preproc_module_config
            )
        elif input_preproc_module_type == "behavior_input":
            self.input_preproc_module = BehaviorInputFeaturesPreprocessor(
                **input_preproc_module_config
            )
        else:
            self.input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
                **input_preproc_module_config
            )

        hstu_module_config = model_args.get("hstu_module_config")
        common_hp = hstu_module_config.get("common_hp")
        model_cfg = hstu_module_config.get("model_cfg")
        self.num_layers = hstu_module_config.get("num_layers")
        model_cls_dict = {}
        self.hstu = FUXI(model_cfg, common_hp, model_cls_dict)

        output_postproc_module_config = model_args.get("output_postproc_module_config")
        output_postproc_module_type = output_postproc_module_config.pop("type")
        if output_postproc_module_type == "l2_norm":
            self.output_postproc_module = L2NormEmbeddingPostprocessor(
                **output_postproc_module_config
            )
        else:
            self.output_postproc_module = LayerNormEmbeddingPostprocessor(
                **output_postproc_module_config
            )

        sampler_module_config = model_args.get("sampler_module_config")
        self.sampler_module_type = sampler_module_config.pop("type", None)
        if self.sampler_module_type == "in_batch":
            self.negatives_sampler = InBatchNegativesSampler(
                **sampler_module_config
            )
        elif self.sampler_module_type == "hyrid_local":
            self.negatives_sampler = HybridLocalNegativesSampler(
                **sampler_module_config
            )
        elif self.sampler_module_type == "local":
            self.negatives_sampler = LocalNegativesSampler(
                num_items=self.embedding_module.num_items,
                item_emb=self.embedding_module._item_emb,
                all_item_ids=[x + 1 for x in range(self.embedding_module.num_items)],
                **sampler_module_config
            )
        else:
            self.negatives_sampler = None
            raise ValueError(f"Unrecognized sampling strategy {self.sampler_module_type}.")

        similarity_module_config = model_args.get("similarity_module_config")
        similarity_module_type = similarity_module_config.pop("type", None)
        if similarity_module_type == "DotProduct":
            self.similarity_module = DotProductSimilarity()
        elif similarity_module_type == "MoL":
            self.similarity_module, _ = create_mol_interaction_module(
                **similarity_module_config
            )
        else:
            self.similarity_module = None
            raise ValueError(f"Unknown interaction_module_type {similarity_module_type}")

        loss_module_config = model_args.get("loss_module_config")
        loss_module_type = loss_module_config.pop("type", "none")
        if loss_module_type == "BCELoss":
            self.loss_module = BCELoss(temperature=loss_module_config.get("temperature"),
                                       similarity_module=self.similarity_module)
        elif loss_module_type == "SampledSoftmaxLoss":
            self.loss_module = SampledSoftmaxLoss(
                **loss_module_config,
                similarity_module=self.similarity_module
            )
        else:
            self.loss_module = None
            raise ValueError(f"Unrecognized loss module {loss_module_type}.")

    def forward(self, past_ids, past_lengths, past_payloads=None):
        input_embeddings = self.embedding_module.get_item_embeddings(past_ids)

        past_lengths, user_embeddings, _ = self.input_preproc_module(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=input_embeddings,
            past_payloads=past_payloads,
        )

        B, L = past_ids.size()
        invalid_attn_mask = torch.triu(torch.ones((B, L, L), dtype=torch.int))
        ffn_output, (v, q, k, ffn_output), time_bias = self.hstu(
            x=user_embeddings,
            x_offsets=None,
            all_timestamps=None,
            invalid_attn_mask=invalid_attn_mask,
            past_lengths=past_lengths,
            num_rerank=0,
            layer_num=self.num_layers
        )

        seq_embeddings = self.output_postproc_module(ffn_output)

        return seq_embeddings

    def get_loss(self, past_ids, past_lengths, seq_embeddings):
        supervision_ids = past_ids

        if self.sampler_module_type == "in_batch":
            in_batch_ids = supervision_ids.view(-1)
            self.negatives_sampler.process_batch(
                ids=in_batch_ids,
                presences=(in_batch_ids != 0),
                embeddings=self.embedding_module.get_item_embeddings(in_batch_ids),
            )
        else:
            if self.side_flag:
                self.negatives_sampler._item_emb = self.embedding_module._item_emb
                self.negatives_sampler._side_info_emb = self.input_features_preproc._item_features_emb
            else:
                self.negatives_sampler._item_emb = self.embedding_module._item_emb

        ar_mask = supervision_ids[:, 1:] != 0
        if self.side_flag:
            supervision_embeddings = self.negatives_sampler.module.id_with_side_emb(supervision_ids[:, 1:])
        else:
            input_embeddings = self.embedding_module.get_item_embeddings(supervision_ids)
            supervision_embeddings = input_embeddings[:, 1:, :]

        loss = self.loss_module(
            lengths=past_lengths - 1,  # [B],
            output_embeddings=seq_embeddings[:, :-1, :],  # [B, N-1, D]
            supervision_ids=supervision_ids[:, 1:],  # [B, N-1]
            supervision_embeddings=supervision_embeddings,  # [B, N - 1, D]
            supervision_weights=ar_mask.float(),
            negatives_sampler=self.negatives_sampler,
        )
        return loss

    def get_user_embedding(self, past_lengths, seq_embeddings, curr=1):
        current_embeddings = get_current_embeddings(lengths=past_lengths, encoded_embeddings=seq_embeddings, curr=curr)
        return current_embeddings

    def get_item_embeddings(self, item_ids=None, turn_id=False):
        if item_ids is None:
            item_embeddings = self.embedding_module._item_emb.weight
        else:
            item_embeddings = self.embedding_module.get_item_embeddings(item_ids)
        norm_item_embeddings = self.negatives_sampler._maybe_l2_norm(item_embeddings)

        if turn_id:
            norm_item_embeddings = norm_item_embeddings.detach().cpu().numpy().tolist()
            id2index = self.embedding_module.get_feature_map()
            origin_ids = []
            new_item_embeddings = []
            for item_id, index in id2index.items():
                origin_ids.append(item_id)
                new_item_embeddings.append(",".join(map(str, norm_item_embeddings[index])))
            return {"item_ids": origin_ids, "item_embeddings": new_item_embeddings}

        return norm_item_embeddings

    def get_topK_logits(self, query_embeddings, top_k=200):
        item_emgbeddings_t = self.get_item_embeddings().t()
        all_logits = torch.mm(query_embeddings, item_emgbeddings_t)
        top_k_logits, top_k_indices = torch.topk(
            all_logits, dim=1, k=top_k, sorted=True, largest=True,
        )  # (B, k,)
        return top_k_indices
