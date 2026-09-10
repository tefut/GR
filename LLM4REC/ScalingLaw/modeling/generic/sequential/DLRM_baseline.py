from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

from modeling.generic.sequential.DLRM_block import AverageLayer
from modeling.model_registry import ModelRegistry
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.deep_modules import DLRModule


@ModelRegistry.register()
class DLRM(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp["feature_conf"]
        model_conf = common_hp["model_conf"]
        item_feature_columns: Dict = feat_conf.get('item_feature_columns', None)
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)
        self.item_feature_columns = item_feature_columns

        self.layers = nn.ModuleDict()

        self.layers["random_embedding_layer"] = self.init_sub_model("embedding_layer_config_random")
        self.layers["llm_embedding_layer"] = self.init_sub_model("embedding_layer_config_llm")

        self.layers["average_layer"] = self.init_sub_model("average_layer")

        for feature in ["play_song_seq", "switch_song_seq", "dislike_song_seq", "profile_song_seq", "positive_song_seq",
                        "pref_song", "pref_song_short"]:
            self.layers[f"srn_{feature}"] = self.init_sub_model("srn")

        for feature in ["play_song_seq", "switch_song_seq", "dislike_song_seq", "profile_song_seq",
                        "positive_song_seq"]:
            self.layers[f"dot_product_attention_{feature}"] = self.init_sub_model("dot_product_attention")

        for feature in ["pref_song", "pref_song_short"]:
            self.layers[f"artist_attention_{feature}"] = self.init_sub_model("attention_artist")

        for feature in ["pref_song", "pref_song_short"]:
            self.layers[f"genre_attention_{feature}"] = self.init_sub_model("attention_genre")

        self.layers["cross_net"] = self.init_sub_model("cross_net")

        self.layers["mlp_net"] = self.init_sub_model("mlp_net")

        self.layers["mmoe"] = self.init_sub_model("mmoe")
        moe_out_dim = self.layers["mmoe"].out_dim

        self.layers["average"] = self.init_sub_model("average")
        self.layers["batch_norm"] = nn.BatchNorm1d(num_features=1)

        self.relu = nn.ReLU()

    def forward(self,
                past_ids,
                num_rerank,
                model_inputs,
                user_feature_embs_original,
                item_feature_embs_original,
                seq_feature_embs):

        inputs = dict(model_inputs)

        # 特征随机向量
        labels = inputs.pop("label", None)
        user_id = inputs.pop("user_id", torch.tensor(0))
        valid_items = inputs.pop("valid_items", torch.tensor(0))
        past_lengths = inputs.pop("past_lengths", None)
        B = inputs.get("song_id").shape[0]
        if torch.onnx.is_in_onnx_export() or not self.training:
            for k, v in inputs.items():
                if v.size(-1) == num_rerank:
                    bsize, seq_len, num_re = v.size()
                    inputs[k] = v.contiguous().permute(0, 2, 1).reshape(bsize * num_re, seq_len)
                else:

                    original_dim = v.dim()  # 1 or 2

                    if original_dim == 1:
                        v = v.unsqueeze(1)  # [B] -> [B, 1]

                    # Now v is [B, D]
                    v_expanded = v.unsqueeze(1).expand(-1, num_rerank, -1)  # [B, num_rerank, D]
                    v_repeated = v_expanded.reshape(-1, v.size(1))  # [B * num_rerank, D]

                    if original_dim == 1:
                        v_repeated = v_repeated.squeeze(1)  # Restore [B * num_rerank]
                    inputs[k] = v_repeated

        random_embedding = self.layers["random_embedding_layer"](inputs)
        llm_embedding = self.layers["llm_embedding_layer"](inputs)

        # 用户长期序列特征加入pos信息，再和对应的llm embedding进行合并；
        seq_feature = {}
        for seq in ["play_song_seq", "switch_song_seq", "dislike_song_seq", "profile_song_seq", "positive_song_seq"]:
            if seq == "dislike_song_seq":
                pos = random_embedding["pos_seq"][:, :100, :]
            else:
                pos = random_embedding["pos_seq"]
            seq_feature[seq] = random_embedding[seq] + pos + llm_embedding[seq]

        for seq in ["pref_song", "pref_song_short", "song_id"]:
            seq_feature[seq] = random_embedding[seq] + llm_embedding[seq]

        # 对item sideinfo进行降维
        item_attr = {}
        for feature in ["genre", "scene", "theme", "mood", "langua", "eras", "artist"]:
            item_attr[feature] = random_embedding[feature]

        # loop srn
        # 生成多个srn操作，执行多次
        srn = []
        for feature in ["play_song_seq", "switch_song_seq", "dislike_song_seq", "profile_song_seq", "positive_song_seq",
                        "pref_song", "pref_song_short"]:
            mask = (inputs[feature] != 0)
            srn.append(self.layers[f"srn_{feature}"](seq_feature.get("song_id"), seq_feature.get(feature), mask))

        # artist dot attention
        artist_dot_attention = []
        for feature in ["play_song_seq", "switch_song_seq", "dislike_song_seq", "profile_song_seq",
                        "positive_song_seq"]:
            mask = [None, (inputs[feature] != 0)]
            artist_dot_attention.append(
                self.layers[f"dot_product_attention_{feature}"](item_attr.get("artist"),
                                                                seq_feature.get(feature),
                                                                mask))

        # artist attention
        artist_attention = []
        for feature in ["pref_song", "pref_song_short"]:
            mask = [None, (inputs[feature] != 0)]
            artist_attention.append(
                self.layers[f"artist_attention_{feature}"](item_attr.get("artist"), seq_feature.get(feature), mask))

        # genre attention
        genre_attention = []
        for feature in ["pref_song", "pref_song_short"]:
            mask = [None, (inputs[feature] != 0)]
            genre_attention.append(
                self.layers[f"genre_attention_{feature}"](item_attr.get("genre"), seq_feature.get(feature), mask))

        artist_dot_attention_concat = torch.concat(artist_dot_attention, dim=-1).squeeze(1)
        artist_attention_concat = torch.concat(artist_attention, dim=-1).squeeze(1)
        genre_attention_concat = torch.concat(genre_attention, dim=-1).squeeze(1)
        # 特征汇总
        attention_concat = torch.concat([artist_attention_concat, artist_dot_attention_concat, genre_attention_concat],
                                        dim=-1)
        srn_concat = torch.concat(srn, dim=-1)
        if srn_concat.dim() == 3:
            srn_concat = srn_concat.squeeze(1)  # 默认会去掉所有为1的维度

        feature_squeeze_list = []
        batch_norm = []
        for k, v in random_embedding.items():
            if v.dim() == 3:
                if v.shape[1] > 1:
                    value = self.layers["average"](v)
                else:
                    value = v
            else:
                batch_norm.append(k)
                value = self.layers["batch_norm"](v.float())
                value = value.unsqueeze(-1)

            feature_squeeze_list.append(value)

        feature_squeeze = torch.cat(feature_squeeze_list, dim=-1).squeeze(1)
        all_feature_concat = torch.cat([feature_squeeze, attention_concat, srn_concat], dim=-1)

        # dcn
        cross_net_output = self.layers[f"cross_net"](all_feature_concat)
        deep_layer_output = self.layers[f"mlp_net"](all_feature_concat)

        deep_cross_concat = torch.cat([cross_net_output, deep_layer_output], dim=-1)

        # mmoe
        mmoe_output = self.layers[f"mmoe"](deep_cross_concat)

        tow_click = mmoe_output[0]

        _, dim = tow_click.size()
        final_logit = tow_click.view(B, num_rerank, dim)

        model_inputs["label"] = labels
        model_inputs["user_id"] = user_id
        model_inputs["valid_items"] = valid_items
        model_inputs["past_lengths"] = past_lengths
        return {"deep_outputs": final_logit, "deep_loss": (0.0, 0.0)}
