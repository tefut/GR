import abc
import logging
import os
from collections import ChainMap
from typing import Dict, List, Tuple, Optional

import torch
from dataclasses import dataclass
from modeling.generic.initialization import truncated_normal
from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from utils.common_utils import read_json


class EmbeddingModule(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

    @abc.abstractmethod
    def get_item_embeddings(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_candidate_item_embeddings(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_user_embeddings(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


@ModelRegistry.register()
class LocalEmbeddingModuleWithSideInfoLonger(EmbeddingModule):
    """
    带有sideinfo的Embedding模块, 用于生成物品和用户的表示。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp["feature_conf"]
        model_conf = common_hp["model_conf"]
        item_feature_columns: Dict = feat_conf.get('item_feature_columns', None)
        self.item_feature_columns = item_feature_columns
        user_feature_columns: Dict = feat_conf.get('user_feature_columns', None)
        self.user_feature_columns = user_feature_columns
        self.sequence_feature_columns: Dict = feat_conf.get('seq_feature_columns', None)
        self.infer_item_id_name = feat_conf.get("infer_items_key")
        self.padding_index = feat_conf.get("padding_index", 0)

            
        # 构造辅助embedding字典
        self.aux_embedding_dict = torch.nn.ModuleDict()
        self.aux_dim = {}
        aux_dim_total = 0
        aux_root_path = model_conf.get("LLM_embedding_dir", None)
        item_feature_count = item_feature_columns[self.infer_item_id_name]["feature_count"]
        feature_map_file = model_conf.get("feature_map_dir_or_path", None)
        auxiliary_config = model_conf.get("auxiliary_embeddings", {})
        if len(auxiliary_config) > 0:
            song_id_featuremap = read_json(feature_map_file)["sparse"][self.infer_item_id_name]
            for auxiliary_name, auxiliary_config in auxiliary_config.items():
                auxiliary_name = auxiliary_name + "_emb_table"
                auxiliary_path = auxiliary_config["path"]
                auxiliary_dim = auxiliary_config["dim"]
                self.aux_dim[auxiliary_name] = auxiliary_dim
                auxiliary_requires_grad = auxiliary_config["requires_grad"]
                auxiliary_path = os.path.join(aux_root_path, auxiliary_path)
                aux_embedding = self.set_aux_embedding(auxiliary_name, auxiliary_path, song_id_featuremap,
                                      item_feature_count, auxiliary_dim, auxiliary_requires_grad)
                self.aux_embedding_dict[auxiliary_name] = aux_embedding
                aux_dim_total = aux_dim_total + auxiliary_dim

        self.multi_value_prefix = feat_conf.get('multi_value_prefix', "pref_")
        if user_feature_columns is None or item_feature_columns is None or self.sequence_feature_columns is None:
            raise ValueError("user_feature_columns and item_feature_columns cannot be None")
        # 物品信息嵌入字典
        _all_info_embs = {}
        self._item_feature_names: List[str] = []
        self.item_dyte: Dict[str, str] = {}
        item_info_dims = {}
        # 用户信息嵌入字典
        self._user_feature_names: List[str] = []
        self.user_dyte: Dict[str, str] = {}
        user_info_dims = {}
        # 序列信息嵌入字典
        self._seq_feature_names: Dict[str, str] = {}
        self.seq_dyte: Dict[str, str] = {}
        seq_info_dims = {}

        # 初始化物品特征嵌入
        self.enabled_item_features: List[str] = []
        for feature_name, _feature_info in item_feature_columns.items():
            feature_count = item_feature_columns[feature_name].get('feature_count', 10)
            feature_enabled = item_feature_columns[feature_name].get("enabled", True)
            feature_dtype = item_feature_columns[feature_name].get("dtype", "int")
            self.item_dyte[feature_name] = feature_dtype
            if feature_enabled == True:
                self.enabled_item_features.append(feature_name)
                if feature_dtype == "con":
                    # 处理商品特征里的离散特征
                    feature_dim = 1
                    _con_layer = torch.nn.BatchNorm1d(1)
                    for p in _con_layer.parameters():
                        p.requires_grad = False
                    _all_info_embs[feature_name] = _con_layer
                elif feature_dtype == "int" or feature_dtype == "multi":
                    feature_dim = item_feature_columns[feature_name].get('dim', 32)
                    _item_emb_table = torch.nn.Embedding(feature_count + 1, feature_dim, padding_idx=self.padding_index)
                    _item_emb_table.weight.requires_grad = False
                    _all_info_embs[feature_name] = _item_emb_table
                else:
                    logging.error("feature_dtype %s is undefined for %s.", feature_dtype, feature_name)
                self._item_feature_names.append(feature_name)
                if _feature_info.get("enable_aux_emb", False):
                    feature_dim = feature_dim + aux_dim_total
                item_info_dims[feature_name] = feature_dim   
                
        for _, layer in _all_info_embs.items():
            for param in layer.parameters():
                param.requires_grad = False

        # 初始化用户特征嵌入
        self.enabled_user_features: List[str] = []
        for feature_name, _feature_info in user_feature_columns.items():
            feature_count = user_feature_columns[feature_name].get('feature_count', 10)
            
            feature_enabled = user_feature_columns[feature_name].get("enabled", True)
            feature_dtype = user_feature_columns[feature_name].get("dtype", "int")
            associated_item_feature = user_feature_columns[feature_name].get("associated", None)
            self.user_dyte[feature_name] = feature_dtype
            if feature_enabled == True:
                self.enabled_user_features.append(feature_name)
                if feature_dtype == "pref":
                    # 默认的关联特征是去掉pref_前缀的商品特征名，例如pref_artist默认的关联特征是artist
                    associated = feature_name.replace(self.multi_value_prefix,
                                                      "") if associated_item_feature is None \
                        else associated_item_feature
                    if associated in self.enabled_item_features:
                        # 若关联特征是item embedding 且启用
                        feature_dim = item_info_dims.get(associated)
                        _all_info_embs[feature_name] = _all_info_embs.get(associated)
                        user_info_dims[feature_name] = feature_dim
                    else:
                        logging.error(
                            "The assoicated feature %s for %s is not an item feature or not enabled in config.",
                            associated, feature_name)
                elif feature_dtype == "int" or feature_dtype == "multi":
                    associated = user_feature_columns[feature_name].get("shared", None)
                    if associated is None:
                        # 若无关联特征，直接初始化
                        feature_dim = user_feature_columns[feature_name].get('dim', 32)
                        if _feature_info.get("enable_aux_emb", False):
                            feature_dim = feature_dim + aux_dim_total
                        _all_info_embs[feature_name] = torch.nn.Embedding(feature_count + 1, feature_dim,
                                                                          padding_idx=self.padding_index)
                        _all_info_embs[feature_name].weight.requires_grad = False
                        user_info_dims[feature_name] = feature_dim
                    else:
                        feature_dim = user_info_dims.get(associated)
                        _all_info_embs[feature_name] = _all_info_embs.get(associated)
                        user_info_dims[feature_name] = feature_dim
                else:
                    logging.error("feature_dtype %s is undefined for the user.", feature_dtype)
                self._user_feature_names.append(feature_name)

        # 初始化序列特征嵌入
        user_item_feature_dim = {**item_info_dims, **user_info_dims}

        # 对每个序列的特征分别处理
        for seq_name, seq_config in self.sequence_feature_columns.items():
            seq_info_dims[seq_name] = {}
            for feature_name, feature_info in seq_config.items():
                feature_enabled = feature_info.get("enabled", True)
                if feature_enabled:
                    feature_dtype = feature_info.get("dtype", "int")
                    if feature_dtype == "pref":
                        # "pos_seq"属于所有序列都共享的特征，只处理第一遍就行，后面直接跳过
                        associated = feature_info.get("associated", None)
                        if associated in _all_info_embs:
                            _all_info_embs[feature_name] = _all_info_embs.get(associated)
                        else:
                            logging.error("assoiciated feature %s never appears before.", associated)
                        
                        seq_info_dims[seq_name][feature_name] = user_item_feature_dim.get(associated)
                    elif feature_dtype in ["int"]:
                        feature_dim = feature_info.get("dim", 32)
                        feature_count = int(feature_info.get("feature_count", 10))
                        table = torch.nn.Embedding(feature_count + 1, feature_dim)
                        if feature_name not in _all_info_embs:
                            _all_info_embs[feature_name] = table
                        if feature_info.get("enable_aux_emb", False):
                            feature_dim = feature_dim + aux_dim_total
                        seq_info_dims[seq_name][feature_name] = feature_dim

        self._item_embedding_dim = common_hp["model_conf"].get("item_embedding_dim", 64)
        self.all_info_embs = torch.nn.ModuleDict(_all_info_embs)
        # 计算物品和用户输入维度
        item_input_dim = sum(item_info_dims.values())
        user_input_dim = sum(user_info_dims.values())
        
        feat_conf["item_emb_dims"] = item_input_dim
        feat_conf["user_emb_dims"] = user_input_dim
        
        seq_input_dim = sum(next(iter(seq_info_dims.values())).values())

        self.seq_emb_mlp = torch.nn.Identity()
        if self._item_embedding_dim != 0:

            self.seq_emb_mlp = torch.nn.Sequential(
                torch.nn.Linear(seq_input_dim, self._item_embedding_dim * 4),
                torch.nn.ReLU(),
                torch.nn.Linear(self._item_embedding_dim * 4, self._item_embedding_dim),
                torch.nn.ReLU()
            )
            logging.info('Set seq_emb_mlp to Linear: %s -> %s' % (seq_input_dim, self._item_embedding_dim))

        else:
            if item_input_dim != user_input_dim:
                raise RuntimeError('item_input_dim and user_input_dim mismatch! user_dim : %s, item_dim : %s' %
                                   (user_input_dim, item_input_dim))
            self._item_embedding_dim = item_input_dim

        self.reset_params()
        
    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim

    def reset_params(self):
        for name, params in self.named_parameters():
            if 'emb' in name:
                logging.info("Initialize %s as truncated normal: %s params", name, params.data.size())
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                logging.info("Skipping initializing params %s - not configured", name)

    def set_aux_embedding(self, name, file_path, song_id_featuremap, item_feature_count, 
                          feature_dim, requires_grad):
        """
        初始化商品的辅助特征（如LLM embedding，多路召回的其他embedding等）
        """
        total_embedding, embedding_hit, feature_dict = 0, 0, {}
        files = os.listdir(file_path)
        for feature_file in files:
            emb_file = os.path.join(file_path, feature_file)
            with open(emb_file, "r") as f:
                for row in f.readlines():
                    total_embedding += 1
                    # mir_song_embedding里每一行结尾处有奇怪的字符，去掉
                    origin_id, feature = "|".join(row.split("|", 2)[:2]).split("|")

                    idx_id = int(song_id_featuremap.get(str(origin_id), 0))
                    if idx_id == 0:
                        continue
                    feature_dict[idx_id] = [float(x) for x in feature.split(",")]
                    embedding_hit += 1
        logging.info("%s: totally %s embeddings avaliable, %s can be found in feature map.",
                     name, total_embedding, embedding_hit)
        emb_table = torch.nn.Embedding(item_feature_count + 1, feature_dim, padding_idx=self.padding_index)
        weight_matrix = torch.rand((item_feature_count + 1, feature_dim))
        for idx, vector in feature_dict.items():
            weight_matrix[idx] = torch.tensor(vector)
        weight_matrix.requires_grad = requires_grad
        emb_table.weight.data.copy_(weight_matrix)
        emb_table.requires_grad_(requires_grad)
        return emb_table
    
    def get_item_embeddings(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        根据物品特征获取物品嵌入。

        :param item_features: 物品特征字典。
        :return: 物品嵌入张量。
        """
        eps = 1e-6
        feature_emb_list = []
        for feature_name in self._item_feature_names:
            if feature_name in self.enabled_item_features:
                item_feature_id = item_features[feature_name]
                if self.item_dyte[feature_name] == "con":
                    # item_feature_id: (B, 1, N) N是候选商品数量
                    if len(item_feature_id.size()) == 2:
                        item_feature_id = item_feature_id.unsqueeze(1)
                    feature_value = self.all_info_embs[feature_name](item_feature_id)
                    feature_value = feature_value.transpose(1, 2)
                elif self.item_dyte[feature_name] == "multi":
                    # item_feature_id： (B, N, M) M 是多值特征padding后的长度
                    if len(item_feature_id.size()) == 2:
                        item_feature_id = item_feature_id.unsqueeze(1)
                    feature_values = self.all_info_embs[feature_name](item_feature_id)
                    feature_dim = feature_values.size(-1)
                    feat_nonzero = item_feature_id != self.padding_index
                    feat_mask = feat_nonzero.unsqueeze(-1).repeat(1, 1, 1, feature_dim)
                    # 对pref多值特征做mean pooling，并忽略掉值为self.padding_index的填充index
                    feature_value = (feature_values * feat_mask).sum(dim=2) / (feat_mask.sum(dim=2) + eps)
                else:
                    if len(item_feature_id.size()) == 3:
                        item_feature_id = item_feature_id.squeeze(1)
                    # item_feature_id: (B, N) N是候选商品数量
                    feature_value = self.all_info_embs[feature_name](item_feature_id)
                
                if self.item_feature_columns[feature_name].get("enable_aux_emb", False):
                    # 如果此特征启用辅助embedding，则将辅助embedding和该特征拼接
                    if self.item_dyte[feature_name] in ["multi"]:
                        aux_embeddings = []
                        for k, v in self.aux_embedding_dict.items():
                            aux_dim = self.aux_dim.get(k)
                            feat_mask = feat_nonzero.unsqueeze(-1).repeat(1, 1, aux_dim)
                            aux_value = v(item_feature_id)
                            aux_value = (aux_value * feat_mask).sum(dim=1) / (feat_mask.sum(dim=1) + eps)
                            aux_embeddings.append(aux_value)
                    else:
                        aux_embeddings = [v(item_feature_id) 
                                          for _, v in self.aux_embedding_dict.items()]
                    aux_embeddings.append(feature_value)
                    feature_value = torch.cat(aux_embeddings, dim=-1)
            
            feature_emb_list.append(feature_value)
        if len(feature_emb_list) > 0:
            feature_embs_original = torch.cat(feature_emb_list, dim=-1)
        else:
            feature_embs_original = None
        
        return feature_embs_original
    
    def get_user_embeddings(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        根据用户特征获取用户嵌入。

        :param user_features: 用户特征字典。
        :return: 用户嵌入张量。
        """
        eps = 1e-6
        feature_emb_list = []
        for feature_name in self._user_feature_names:
            if feature_name in self.enabled_user_features:
                user_feature_id = user_features[feature_name]
                if self.user_dyte[feature_name] == "pref" or self.user_dyte[feature_name] == "multi":
                    # feature_values: (B, M, D) M 是多值特征padding后的长度
                    feature_values = self.all_info_embs[feature_name](user_feature_id)
                    feature_dim = feature_values.size(-1)
                    feat_nonzero = user_feature_id != self.padding_index
                    feat_mask = feat_nonzero.unsqueeze(-1).repeat(1, 1, feature_dim)
                    # 对pref多值特征做mean pooling，并忽略掉值为self.padding_index的填充index
                    feature_value = (feature_values * feat_mask).sum(dim=1) / (feat_mask.sum(dim=1) + eps)
                else:
                    feature_value = self.all_info_embs[feature_name](user_feature_id)
                if self.user_feature_columns[feature_name].get("enable_aux_emb", False):
                    # 如果此特征启用辅助embedding，则将辅助embedding和该特征拼接
                    if self.user_dyte[feature_name] in ["pref", "multi"]:
                        aux_embeddings = []
                        for k, v in self.aux_embedding_dict.items():
                            aux_dim = self.aux_dim.get(k)
                            feat_mask = feat_nonzero.unsqueeze(-1).repeat(1, 1, aux_dim)
                            aux_value = v(user_feature_id)
                            aux_value = (aux_value * feat_mask).sum(dim=1) / (feat_mask.sum(dim=1) + eps)
                            aux_embeddings.append(aux_value)
                    else:
                        aux_embeddings = [v(user_feature_id) 
                                          for _, v in self.aux_embedding_dict.items()]
                    aux_embeddings.append(feature_value)
                    feature_value = torch.cat(aux_embeddings, dim=-1)
            feature_emb_list.append(feature_value)
        if len(feature_emb_list) > 0:
            feature_embs_original = torch.cat(feature_emb_list, dim=-1)
        else:
            feature_embs_original = None
        
        return feature_embs_original

    def get_seq_embeddings(self, seq_features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        all_seq_embs = []
        for _, seq_config in self.sequence_feature_columns.items():
            feature_emb_list = []
            for feature_name, feature_info in seq_config.items():
                seq_feature_id = seq_features[feature_name]
                S = feature_info.get("length")
                feature_value = self.all_info_embs[feature_name](seq_feature_id)[:, :S, :]
                if feature_info.get("enable_aux_emb", False):
                    aux_embeddings = [v(seq_feature_id) for _, v in self.aux_embedding_dict.items()]
                    aux_embeddings.append(feature_value)
                    feature_value = torch.cat(aux_embeddings, dim=-1)
                feature_emb_list.append(feature_value)
            seq_emb = torch.cat(feature_emb_list, dim=-1)
            all_seq_embs.append(seq_emb)
        feature_embs = torch.cat(all_seq_embs, dim=1)
        if self._item_embedding_dim != 0:
            feature_embs = self.seq_emb_mlp(feature_embs)
        return feature_embs
