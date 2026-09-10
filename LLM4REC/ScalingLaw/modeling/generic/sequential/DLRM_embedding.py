import time
import logging
import numpy as np
import json
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
from modeling.generic.sequential.base_model import BaseModel


@ModelRegistry.register()
class FeatureEmbedding(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.exclude_features = model_cfg[Const.HP].get("exclude_features", None)
        self.embedding_layer_config = model_cfg[Const.HP].get("embedding_layer_config", None)
        self.share_embedding_layer = model_cfg[Const.HP].get("share_embedding_layer", None)
        feature_map_file = model_cfg[Const.HP].get("feature_map_file", None)
        self.feature_map, self.feature_vocab_size = self._init_feature_map(feature_map_file)

        self.embedding_layer_dict = nn.ModuleDict()
        embedding_layer_dict = self._init_embedding_layers_dict(self.embedding_layer_config)
        for feature_name, embedding in embedding_layer_dict.items():
            self.embedding_layer_dict[feature_name] = embedding

        for share_feature_name, feature_name_list in self.share_embedding_layer.items():
            for feature_name in feature_name_list:
                if feature_name in self.exclude_features:
                    continue
                self.embedding_layer_dict[feature_name] = embedding_layer_dict.get(share_feature_name)

        # 减少模型导出的数据量
        self.feature_map = {}

    def _init_feature_map(self, feature_map_file):
        with open(feature_map_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
        feature_map = data.get("sparse", {})
        feature_vocab_size = data.get("maxIndexMap", {})
        return feature_map, feature_vocab_size

    def _init_embedding_layers_dict(self, embedding_layer_config):
        embedding_layer_dict = {}
        for embedding_config in embedding_layer_config:
            embedding_config = self.get_config(embedding_config)
            feature_name_list = embedding_config.get("feature_name_list")
            embedding_layer_type = embedding_config.get("type")
            for feature_name in feature_name_list:
                if embedding_layer_type == "embedding":
                    embedding_layer = self._init_embedding_layer(feature_name, embedding_config)
                elif embedding_layer_type == "pretrained_embedding":
                    embedding_layer = self._init_pretrained_embedding_layer(feature_name, embedding_config)
                else:
                    logging.info(f"embedding_layer_type: {embedding_layer_type} is error")
                    embedding_layer = None
                embedding_layer_dict[feature_name] = embedding_layer
        return embedding_layer_dict

    def _init_embedding_layer(self, feature_name, config):
        if feature_name not in self.feature_vocab_size:
            raise RuntimeError(f"{feature_name} not in feature_map.")
        vocab_size = self.feature_vocab_size.get(feature_name)
        embed_dim = config.get("output_dim")
        embeddings_initializer_type = config.get("initializer", "uniform")
        embedding_layer = nn.Embedding(vocab_size + 1, embed_dim)

        # embedding_layer参数初始化
        self.embedding_initializer(embeddings_initializer_type, embedding_layer.weight)

        trainable = config.get("trainable")
        if not trainable:
            for param in embedding_layer.parameters():
                param.requires_grad = False

        return embedding_layer

    def _init_pretrained_embedding_layer(self, feature_name, config):

        pretrained_embedding_file = config.get("pretrained_embedding_file")
        valid_feature_embedding_list, valid_feature_index_list = \
            self._load_pretrain_embedding(feature_name, pretrained_embedding_file)
        vocab_size = self.feature_vocab_size.get(feature_name)
        embedding_dim = len(valid_feature_embedding_list[0])

        embedding_layer = nn.Embedding(vocab_size + 1, embedding_dim)
        embeddings_initializer_type = config.get("initializer", "uniform")

        # embedding_layer参数初始化
        self.embedding_initializer(embeddings_initializer_type, embedding_layer.weight)
        with torch.no_grad():
            embedding_layer.weight[valid_feature_index_list] = torch.tensor(valid_feature_embedding_list)
        trainable = config.get("trainable")
        if not trainable:
            for param in embedding_layer.parameters():
                param.requires_grad = False

        return embedding_layer

    def _load_pretrain_embedding(self, feature_name, pretrained_embedding_file):
        valid_feature_embedding_list = []
        valid_feature_index_list = []
        feature_index = self.feature_map.get(feature_name, {})
        with open(pretrained_embedding_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                item_id, embedding = line.strip().split("|")
                if item_id not in feature_index:
                    continue
                embedding = np.fromstring(embedding, dtype=float, sep=',').tolist()
                valid_feature_embedding_list.append(embedding)
                valid_feature_index_list.append(feature_index[item_id])
        return valid_feature_embedding_list, valid_feature_index_list

    def embedding_initializer(self, name, tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        if name == "uniform":
            nn.init.uniform_(tensor)
        elif name == "normal":
            nn.init.normal_(tensor)
        elif name == "truncated_normal":
            with torch.no_grad():
                size = tensor.shape
                tmp = tensor.new_empty(size + (4,)).normal_()
                valid = (tmp < b) & (tmp > a)
                ind = valid.max(-1, keepdim=True)[1]
                tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
                tensor.data.mul_(std).add_(mean)
        else:
            logging.info("the embeddings initializer not in candidates")

    def get_config(self, user_config):
        default_config = {
            'type': 'embedding',
            'output_dim': 8,
            'embeddings_initializer': 'uniform',
            'embeddings_regularizer': None,
            'activity_regularizer': None,
            'embeddings_constraint': None,
            "feature_name_list": [],
            "pretrained_embedding_file": "",
            "trainable": True
        }
        for key in user_config:
            if key in default_config:
                default_config[key] = user_config[key]
        return default_config

    def forward(self, inputs: dict):
        '''输入batch dict数据，输出embedding'''
        outputs = {}
        for feature_name, input_tensor in inputs.items():
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(1)
            if feature_name in self.embedding_layer_dict:
                outputs[feature_name] = self.embedding_layer_dict[feature_name](input_tensor)
            else:
                outputs[feature_name] = input_tensor
        return outputs
