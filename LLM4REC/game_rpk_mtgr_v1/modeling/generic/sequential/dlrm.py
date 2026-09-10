from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch_npu

from modeling.generic.initialization import truncated_normal
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.dlrm_modules import (CrossNetwork, PPNetLayer, MLPLayer,
                                                      DeepNetwork, NormDropoutLayer, BaselinePPNet)
from modeling.generic.sequential.embedding_modules import EmbeddingModule
from modeling.generic.utils.constants import FeatConst
from modeling.model_registry import ModelRegistry


@ModelRegistry.register()
class DLRM(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg, common_hp, model_cls_dict)
        model_hp = model_cfg.get('hp')
        self.feature_conf = common_hp.get("feature_conf")
        self.feature_groups = self.feature_conf.get("feature_groups")
        model_conf = common_hp.get("model_conf")
        self.amp_dtype = model_conf.get("amp_dtype", "fp16")
        # use feature embedding from sequential model
        self.embedding_module: EmbeddingModule = model_cfg['embedding_module']
        self.seq_input_dim = model_cfg['item_embedding_dim']
        if model_conf.get('use_user_embeddings_for_rerank', False):
            self.seq_input_dim *= 2

        self.use_dlrm_can = model_hp.get('use_dlrm_can', True)
        self.can_as_dcn_input = model_hp.get('can_as_dcn_input', True)
        self.can_as_dnn_input = model_hp.get('can_as_dnn_input', True)
        self.can_as_final_input = model_hp.get('can_as_final_input', True)
        self.use_dlrm_linear = model_hp.get('use_dlrm_linear', True)
        self.use_dlrm_cross = model_hp.get('use_dlrm_cross', True)
        self.use_dlrm_dnn = model_hp.get('use_dlrm_dnn', True)
        self.use_dlrm_ppnet = model_hp.get('use_dlrm_ppnet', True)

        self.seq_as_cross_input = model_hp.get('seq_as_cross_input', True)
        self.seq_as_dnn_input = model_hp.get('seq_as_dnn_input', True)
        self.seq_as_ppnet_input = model_hp.get('seq_as_ppnet_input', True)
        self.seq_as_final_input = model_hp.get('seq_as_final_input', True)

        self.use_hstu = model_conf.get('use_hstu', False)
        self.use_baseline_dcn = model_conf.get('use_baseline_dcn', False)
        self.use_baseline_ppnet = model_conf.get('use_baseline_ppnet', False)

        self.cand_items_key = self.feature_conf.get("candidate_items_key", FeatConst.DFLT_CAND_ITEM_KEY)
        self._feature_meta_cache: Dict[Tuple[str, ...], List[Tuple[str, str, str]]] = {}
        self.candidate_dlrm_features = self.feature_groups['candidate_dlrm']['features']
        self.candidate_dlrm_feature_meta = self._get_feature_meta(self.candidate_dlrm_features)

        # get input dim
        self.cand_feature_dim = self._calculate_input_dim(self.candidate_dlrm_features)

        # init feature interaction modules
        ## CAN
        if self.use_dlrm_can:
            self.can_input_dim = model_hp.get('can_input_dim', 16)
            self.can_hidden_dims = model_hp.get('can_hidden_dims', [8, 4])
            self.can_selected_features = model_hp.get('can_selected_features',
                                                      self.feature_groups['candidate_dlrm']['features'])
            self.cand_can_dim = self._calculate_can_input_dim(self.can_selected_features)
            self.emb_can_table_con = torch.nn.ModuleDict()
            self.emb_can_table_single = torch.nn.ModuleDict()
            self.emb_can_table_multi = torch.nn.ModuleDict()
            self._initialize_features_can_embeddings(self.can_selected_features)

        ## Cross
        if self.use_dlrm_cross:
            self.num_cross_layers = model_hp.get('num_cross_layers', 3)
            input_dim_cross = self.cand_feature_dim
            if self.use_dlrm_can and self.can_as_dcn_input:
                input_dim_cross += self.cand_can_dim
            if self.seq_as_cross_input:
                input_dim_cross += self.seq_input_dim
            self.cross = CrossNetwork(input_dim_cross, self.num_cross_layers)

        ## DNN
        if self.use_dlrm_dnn:
            self.dnn_hidden_layers = model_hp.get('dnn_hidden_layers', [1024, 512, 256])
            input_dim_dnn = self.cand_feature_dim
            if self.use_dlrm_can and self.can_as_dnn_input:
                input_dim_dnn += self.cand_can_dim
            if self.seq_as_dnn_input:
                input_dim_dnn += self.seq_input_dim
            if self.use_baseline_dcn:
                # dnn前的norm_dropout激活
                self.norm_before_dnn = NormDropoutLayer(
                    in_dim=input_dim_dnn,
                    activation=model_hp.get('norm_before_dnn_act', "relu"),
                    order=model_hp.get('norm_before_dnn_order', 'a')
                )
                # 使用基线对齐的Deep网络
                self.dnn = DeepNetwork(
                    hidden_dims=[input_dim_dnn] + self.dnn_hidden_layers,
                    hidden_activations=model_hp.get('dnn_act_func', 'relu'),
                    dropout_rates=model_hp.get('dnn_dropout_rate', 0.1),
                    norms=model_hp.get('dnn_use_bn', False),
                    l2_reg=model_hp.get('l2_reg', 1e-5),
                    orders=model_hp.get('dnn_orders', "nad")
                )
                # dnn后的norm_dropout激活
                self.norm_after_dnn = NormDropoutLayer(
                    in_dim=self.dnn_hidden_layers[-1],
                    norm_type=model_hp.get('norm_type', "batch_norm"),
                    dropout_rate=model_hp.get('norm_before_dnn_dropout_rate', 0.1),
                    order=model_hp.get('norm_after_dnn_order', "nd")
                )
            else:
                self.dnn = MLPLayer(
                    hidden_layers=[input_dim_dnn] + self.dnn_hidden_layers,
                    act_func=model_hp.get('dnn_act_func', 'relu'),
                    dropout_rate=model_hp.get('dnn_dropout_rate', 0.1),
                    use_bn=model_hp.get('dnn_use_bn', False)
                )

        ## linear
        if self.use_dlrm_linear:
            self.dlrm_linear = torch.nn.Linear(self.cand_feature_dim, 1, bias=False)

        ## PPNet
        if self.use_dlrm_ppnet:
            self.ppnet_gate_features = model_hp.get('ppnet_gate_features',
                                                    self.feature_groups['candidate_dlrm']['features'])
            self.input_ppnet_gate_dim = self._calculate_ppnet_gate_input_dim()
            self.ppnet_mlp_layer_dims = model_hp.get('ppnet_mlp_layer_dims', [256, 128])
            self.ppnet_gate_hidden_dims = model_hp.get('ppnet_gate_hidden_dims',
                                                       [self.input_ppnet_gate_dim // 2, self.input_ppnet_gate_dim // 2])
            self.ppnet_dropout_rate = model_hp.get('ppnet_dropout_rate', 0.1)
            self.ppnet_use_bn = model_hp.get('ppnet_use_bn', True)

            input_dim_ppnet = self.cand_feature_dim
            if self.seq_as_ppnet_input:
                input_dim_ppnet += self.seq_input_dim
            self.ppnet = PPNetLayer(
                feature_emb_dim=input_dim_ppnet,  # 特征嵌入维度
                gate_emb_dim=self.input_ppnet_gate_dim,  # 门控嵌入维度
                mlp_layer_dims=self.ppnet_mlp_layer_dims,  # MLP 层维度
                gate_layer_dims=self.ppnet_gate_hidden_dims,  # 门控层维度
                batch_norm=self.ppnet_use_bn,
                dropout_rate=self.ppnet_dropout_rate
            )

        if self.use_baseline_ppnet:
            self.target_features = model_conf.get('target_features', [])
            logging.info("target_features: %s", self.target_features)
            if not self.target_features:
                raise ValueError("target_features must be configured when use_baseline_ppnet is enabled")

            self.domain_gate_features = (
                    self.feature_groups['candidate']['features'] +
                    self.feature_groups['user']['features']
            )
            self.domain_no_target_features = [
                f for f in self.domain_gate_features
                if f not in self.target_features
            ]
            self.domain_gate_input_dim = self._calculate_input_dim(self.domain_gate_features)
            self.domain_no_target_input_dim = self._calculate_input_dim(self.domain_no_target_features)
            self.domain_target_input_dim = self._calculate_input_dim(self.target_features)
            self.domain_gate_units = model_hp.get('baseline_domain_gate_units', self.domain_target_input_dim)
            self.baseline_ppnet_output_dim = self.domain_no_target_input_dim + model_hp.get(
                'baseline_hidden_dims',
                model_hp.get('dnn_hidden_layers', [1024, 512, 256, 128])
            )[-1]

            self.baseline_ppnet = BaselinePPNet(
                hidden_dims=model_hp.get('baseline_hidden_dims',
                                         model_hp.get('dnn_hidden_layers', [1024, 512, 256, 128])),
                domain_gate_input_dim=self.domain_gate_input_dim,
                domain_no_target_input_dim=self.domain_no_target_input_dim,
                target_input_dim=self.domain_target_input_dim,
                domain_gate_units=self.domain_gate_units,
                num_cross_layers=model_hp.get('num_cross_layers', 3),
                dropout_rate=model_hp.get('baseline_dropout_rate', 0.1)
            )

        ## final linear
        self.final_input_dim = 0
        if self.use_dlrm_linear:
            self.final_input_dim += 1
        if self.use_dlrm_can and self.can_as_final_input:
            self.final_input_dim += self.cand_can_dim
        if self.use_dlrm_cross:
            self.final_input_dim += input_dim_cross
        if self.use_dlrm_dnn:
            self.final_input_dim += self.dnn_hidden_layers[-1]
        if self.use_dlrm_ppnet:
            self.final_input_dim += self.ppnet_mlp_layer_dims[-1]
        if self.use_baseline_ppnet:
            self.final_input_dim += self.baseline_ppnet_output_dim
        if self.seq_as_final_input:
            self.final_input_dim += self.seq_input_dim

        self.reset_params()

    def reset_params(self):
        for name, module in self.named_modules():
            if 'embedding_module' in name:
                continue
            if isinstance(module, torch.nn.Embedding):
                truncated_normal(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    torch.nn.init.constant_(module.weight.data[module.padding_idx], 0.)
                logging.info(f"Initialize module {name} as truncated normal: {module.weight.data.size()} params")

            elif isinstance(module, torch.nn.Linear):
                if name.startswith('baseline_ppnet'):
                    torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                else:
                    module.weight.data.normal_(mean=0.0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
                logging.info(f"Initialize module {name}")

            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
                logging.info(f"Initialize torch.nn.LayerNorm {name}")

        # for nn.Parameter
        for name, param in self.named_parameters():
            all_module_names = [module_name for module_name, _ in self.named_modules() if module_name]
            if not (name.removesuffix('.weight') in all_module_names or name.removesuffix('.bias') in all_module_names):
                param.data.normal_(mean=0.0, std=0.01)

    def _calculate_input_dim(self, feature_names: List) -> int:
        input_dim = 0
        for _, feat_dtype, base_feat_name in self._get_feature_meta(feature_names):
            if feat_dtype == "con":
                input_dim += 1
            else:
                input_dim += self.embedding_module.emb_table[base_feat_name].weight.shape[1]

        return input_dim

    def _get_feature_meta(self, feature_names: List[str]) -> List[Tuple[str, str, str]]:
        cache_key = tuple(feature_names)
        feature_meta = self._feature_meta_cache.get(cache_key)
        if feature_meta is None:
            feature_meta = [
                (
                    feat_name,
                    self.embedding_module.dtypes.get(feat_name),
                    self.embedding_module._get_base_feature_name(feat_name),
                )
                for feat_name in feature_names
            ]
            self._feature_meta_cache[cache_key] = feature_meta
        return feature_meta

    @staticmethod
    def _align_feature_to_candidates(feature_value: torch.Tensor, num_rerank: int) -> torch.Tensor:
        if feature_value.dim() == 2:
            feature_value = feature_value.unsqueeze(1)

        cur_dim = feature_value.size(1)
        if cur_dim == num_rerank:
            return feature_value
        if cur_dim == 1:
            return feature_value.expand(-1, num_rerank, -1)
        if cur_dim < num_rerank:
            pad_size = num_rerank - cur_dim
            padding = feature_value.new_zeros(feature_value.size(0), pad_size, feature_value.size(2))
            return torch.cat([padding, feature_value], dim=1)
        return feature_value

    @staticmethod
    def _concat_features(feature_embs: List[torch.Tensor], log_prefix: str) -> torch.Tensor:
        if len(feature_embs) == 1:
            return feature_embs[0]
        try:
            return torch.cat(feature_embs, dim=-1)
        except Exception:
            for i, emb in enumerate(feature_embs):
                logging.info("%s %s shape is: %s", log_prefix, i, emb.shape)
            raise

    def _calculate_can_input_dim(self, feature_names: List) -> int:
        input_dim = 0
        num_one_hot = 0
        num_multi_hot = 0
        for feat_name in feature_names:
            feat_dtype = self.embedding_module.dtypes.get(feat_name)
            if feat_dtype == "con":
                input_dim += 1
            elif feat_dtype == "int" or feat_dtype == "context":
                num_one_hot += 1
            elif feat_dtype == "multi":
                # 从feature配置中获取max_len
                feature_columns = ["candidate_item_feature_columns", "history_item_feature_columns",
                                   "user_feature_columns"]
                max_len = 1
                for conf_key in feature_columns:
                    feature_conf = self.feature_conf.get(conf_key, {})
                    if feat_name in feature_conf:
                        max_len = feature_conf[feat_name].get('max_len', 1)
                        break
                num_multi_hot += max_len
        if num_one_hot > 0 or num_multi_hot > 0:
            input_dim += num_one_hot * num_multi_hot * sum(self.can_hidden_dims)
        else:
            # 如果没有有效的CAN特征组合，返回0
            input_dim += 0
        return input_dim

    def _initialize_features_can_embeddings(self, feature_names: list[str]) -> None:
        # all feature_columns
        all_feature_columns = ["candidate_item_feature_columns", "history_item_feature_columns", "user_feature_columns"]

        # single feature dim
        single_emb_dim = 0
        for in_dim, out_dim in zip(([self.can_input_dim] + self.can_hidden_dims)[:-1],
                                   ([self.can_input_dim] + self.can_hidden_dims)[1:]):
            single_emb_dim += in_dim * out_dim

        for feature_name in feature_names:
            for feature_column in all_feature_columns:
                group_feature_conf: dict = self.feature_conf.get(feature_column, None)
                if group_feature_conf is None:
                    return
                if feature_name in group_feature_conf:
                    feature_info = group_feature_conf[feature_name]
                    base_feature_name = self.embedding_module._get_base_feature_name(feature_name)

                    feature_count = feature_info.get('feature_count', FeatConst.FEAT_CNT)
                    feature_enabled = feature_info.get("enabled", True)
                    feature_dtype = feature_info.get("dtype", FeatConst.DFLT_DTYPE)

                    if feature_enabled:
                        if feature_dtype == "con":
                            if base_feature_name in self.emb_can_table_con:
                                pass
                            else:
                                layer = torch.nn.BatchNorm1d(1)
                                self.emb_can_table_con[base_feature_name] = layer
                        elif feature_dtype == "multi":
                            if base_feature_name in self.emb_can_table_multi:
                                pass
                                # use shared feature name
                            shared_feat_name = feature_info.get("shared_feat_name", "")
                            if shared_feat_name not in self.embedding_module.all_feature_names:
                                raise ValueError("multifeat %s, shared_feat_name \"%s\" not found in feature columns",
                                                 feature_name, shared_feat_name)
                            base_shared_feature_name = self.embedding_module._get_base_feature_name(shared_feat_name)
                            if base_shared_feature_name in self.emb_can_table_multi:
                                layer = self.emb_can_table_multi[base_shared_feature_name]
                            else:
                                feature_count = group_feature_conf.get(shared_feat_name) \
                                    .get('feature_count', FeatConst.FEAT_CNT)
                                layer = torch.nn.Embedding(feature_count + 1, self.can_input_dim,
                                                           padding_idx=self.embedding_module.padding_index)
                                self.emb_can_table_multi[base_shared_feature_name] = layer
                            self.emb_can_table_multi[base_feature_name] = layer
                        elif feature_dtype == "int" or feature_dtype == "context":
                            if base_feature_name in self.emb_can_table_single:
                                pass
                            layer = torch.nn.Embedding(feature_count + 1, single_emb_dim,
                                                       padding_idx=self.embedding_module.padding_index)
                            self.emb_can_table_single[base_feature_name] = layer
                        else:
                            logging.error("feature_dtype %s is undefined for %s.", feature_dtype, feature_name)
                            pass
                    break

    def _calculate_ppnet_gate_input_dim(self) -> int:
        input_dim = 0

        gate_features = set(self.ppnet_gate_features)
        for feat_name, feat_dtype, base_feat_name in self.candidate_dlrm_feature_meta:
            if feat_name in gate_features:
                if feat_dtype == "con":
                    input_dim += 1
                elif feat_dtype == "int" or feat_dtype == 'context':
                    input_dim += self.embedding_module.emb_table[base_feat_name].weight.shape[1]
                elif feat_dtype == "multi":
                    input_dim += self.embedding_module.emb_table[base_feat_name].weight.shape[1]
        return input_dim

    def _get_feature_embeddings(
            self,
            model_input: Dict[str, torch.Tensor],
            feature_names: List[str]
    ) -> torch.Tensor:
        """
        model_input: Dict of [B, L] or [B, L, M] or [B, C] or [B, C, M] or [B]
        """
        feature_emb_list = self._get_feature_embedding_list(model_input, feature_names)
        feature_embs = torch.cat(feature_emb_list, dim=-1)
        return feature_embs

    def _get_feature_embedding_list(
            self,
            model_input: Dict[str, torch.Tensor],
            feature_names: List[str]
    ) -> List[torch.Tensor]:
        num_rerank = model_input.get(self.cand_items_key).shape[1]
        feature_emb_list = []
        for feature_name, feature_dtype, base_feature_name in self._get_feature_meta(feature_names):
            feature_id = model_input[feature_name]
            if feature_dtype == "con":
                if feature_id.dim() == 1:
                    feature_id = feature_id.unsqueeze(-1)
                    feature_id_for_bn = feature_id.unsqueeze(-1)
                elif feature_id.dim() == 2:
                    feature_id_for_bn = feature_id.unsqueeze(1)
                else:
                    feature_id_for_bn = feature_id.transpose(1, 2)
                feature_value = self.embedding_module.emb_table[base_feature_name](feature_id_for_bn)
                feature_value = feature_value.transpose(1, 2)
            else:
                feature_value = self.embedding_module.emb_table[base_feature_name](feature_id)
                if feature_value.ndim == 4:
                    mask = (feature_id != self.embedding_module.padding_index).unsqueeze(-1)
                    # 先除再sum：每个元素先除以valid_count再求和，部分和≈mean量级，FP16也不会溢出
                    mask = mask.to(feature_value.dtype)
                    valid_count = mask.sum(dim=2, keepdim=True)
                    feature_value = (feature_value * mask / (valid_count + 1)).sum(dim=2)
            feature_value = self._align_feature_to_candidates(feature_value, num_rerank)
            feature_emb_list.append(feature_value)
        return feature_emb_list

    def _get_feature_embedding_stack(
            self,
            model_input: Dict[str, torch.Tensor],
            feature_names: List[str]
    ) -> torch.Tensor:
        """
        Return feature embeddings as [B, num_rerank, num_features, emb_dim].
        This keeps the feature axis for baseline domain gate broadcasting.
        """
        feature_emb_list = self._get_feature_embedding_list(model_input, feature_names)
        feature_dims = set()
        for feature_value in feature_emb_list:
            feature_dims.add(feature_value.shape[-1])

        if len(feature_dims) != 1:
            raise ValueError(f"BaselinePPNet requires same embedding dim for all gated features, got {feature_dims}")
        return torch.stack(feature_emb_list, dim=-2)

    def _get_baseline_ppnet_embeddings(
            self,
            model_input: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_emb_list = self._get_feature_embedding_list(model_input, self.domain_gate_features)
        target_features = set(self.target_features)
        concat_domain_embedding = torch.cat(feature_emb_list, dim=-1)

        no_target_emb_list = []
        target_emb_list = []
        no_target_dims = set()
        for feature_name, feature_emb in zip(self.domain_gate_features, feature_emb_list):
            if feature_name in target_features:
                target_emb_list.append(feature_emb)
            else:
                no_target_emb_list.append(feature_emb)
                no_target_dims.add(feature_emb.shape[-1])

        if not no_target_emb_list or not target_emb_list:
            raise ValueError("BaselinePPNet requires both target and non-target features")

        concat_domain_no_target_embedding = torch.stack(no_target_emb_list, dim=-2)
        domain_type1_embedding = torch.cat(target_emb_list, dim=-1)
        return concat_domain_embedding, concat_domain_no_target_embedding, domain_type1_embedding

    def _get_can_embeddings(self, model_input: Dict[str, torch.Tensor], feature_names: list[str]) -> torch.Tensor:
        '''
        model_input: Dict of [B, L] or [B, L, M] or [B, C] or [B, C, L] or [B, C, M] or [B]
        '''
        emb_list_con = []
        emb_list_single = []
        emb_list_multi = []

        for feature_name, feature_dtype, base_feature_name in self._get_feature_meta(feature_names):
            feature_id = model_input[feature_name]
            if feature_dtype == "con":
                # 处理连续特征（保持原逻辑不变）
                if feature_id.dim() == 1:
                    feature_id = feature_id.unsqueeze(-1)
                    feature_id_for_bn = feature_id.unsqueeze(-1)
                elif feature_id.dim() == 2:
                    feature_id_for_bn = feature_id.unsqueeze(1)  # [bs, num_rerank]
                else:
                    feature_id_for_bn = feature_id.transpose(1, 2)
                try:
                    feature_value = self.emb_can_table_con[base_feature_name](feature_id_for_bn)
                    feature_value = feature_value.transpose(1, 2)
                    emb_list_con.append(feature_value)
                except Exception as e:
                    logging.info("feature_id_for_bn shape is %s, feature_id_for_bn dim is %s", feature_id_for_bn.shape,
                                 feature_id_for_bn.dim())
                    logging.error("Failed to compute embedding for %s. Exception: %s", feature_name, e)
                    raise e
            elif feature_dtype == "multi":
                multi_emb = self.emb_can_table_multi[base_feature_name](feature_id)  # [B, C, L, D]
                mask = (feature_id != self.embedding_module.padding_index)
                mask = mask.to(multi_emb.dtype).unsqueeze(-1)  # mask保持同dtype，0/1乘法不影响精度
                multi_emb = multi_emb * mask  # [B, C, L, D]
                emb_list_multi.append(multi_emb)
            elif feature_dtype == "int" or feature_dtype == "context":
                single_emb = self.emb_can_table_single[base_feature_name](feature_id)  # [B, C, 16*8*4]
                emb_list_single.append(single_emb)
            else:
                pass

        if emb_list_con:
            con_embs = torch.cat(emb_list_con, dim=-1)
            bs = con_embs.shape[0]
            num_rerank = con_embs.shape[1]
            con_embs = con_embs.view(bs * num_rerank, -1)  # [256,56]
        else:
            con_embs = None

        if emb_list_multi:
            multi_embs = torch.cat(emb_list_multi, dim=-2)  # [B, C, L_t, D]
        else:
            multi_embs = None

        if emb_list_single:
            single_embs = torch.stack(emb_list_single, dim=-2)  # [B, C, N_s, 16*8*4]
        else:
            single_embs = None

        # 处理空tensor的情况
        if multi_embs is None or single_embs is None:
            sample_tensor = model_input[self.cand_items_key]
            return sample_tensor.new_zeros(sample_tensor.shape[0], sample_tensor.shape[1], 0, dtype=sample_tensor.dtype)

        # CAN
        out_seq = []
        can_all_layers = [self.can_input_dim] + self.can_hidden_dims
        cur_idx = 0
        hidden_out = multi_embs
        for i, (in_dim, out_dim) in enumerate(zip(can_all_layers[:-1], can_all_layers[1:])):
            mlp_w = single_embs[..., cur_idx: cur_idx + in_dim * out_dim]  # [B, C, N_s, in_dim*out_dim]
            cur_idx = cur_idx + in_dim * out_dim
            mlp_w = mlp_w.reshape(mlp_w.shape[:-1] + (in_dim, out_dim))  # [B, C, N_s, in_dim, out_dim]
            # CAN einsum的内积求和在FP16下可能溢出，局部保护
            # BF16有与FP32相同的指数范围，内积求和不会溢出，无需禁用autocast
            if hidden_out.dtype == torch.float16:
                with torch.npu.amp.autocast(enabled=False):
                    hidden_out_fp32 = hidden_out.float()
                    mlp_w_fp32 = mlp_w.float()
                    if i == 0:
                        hidden_out_fp32 = torch.einsum('atik,atjkl->atijl', hidden_out_fp32, mlp_w_fp32)
                    else:
                        hidden_out_fp32 = torch.einsum('atijk,atjkl->atijl', hidden_out_fp32, mlp_w_fp32)
                hidden_out = hidden_out_fp32.to(hidden_out.dtype)
            else:
                if i == 0:
                    hidden_out = torch.einsum('atik,atjkl->atijl', hidden_out, mlp_w)
                else:
                    hidden_out = torch.einsum('atijk,atjkl->atijl', hidden_out, mlp_w)
            hidden_out = torch.tanh(hidden_out)
            out_seq.append(hidden_out)

        out_seq = torch.cat(out_seq, dim=-1)
        out_seq = out_seq.reshape(out_seq.shape[0], out_seq.shape[1], -1)  # [B, C, L_t * N_s * (8 + 4)]

        if con_embs is not None:
            batch_size, cand_size = out_seq.shape[0], out_seq.shape[1]
            con_embs_shaped = con_embs.view(batch_size, cand_size, -1)
            out_seq = torch.cat([out_seq, con_embs_shaped], dim=-1)  # [B, C, D_can + D_con]

        return out_seq

    def forward(self, model_input: Dict[str, torch.Tensor], encoded_embedding: torch.Tensor = None):
        """
        修改后的forward函数，适配encoded_embedding作为输入
        model_input: 候选特征字典
        encoded_embedding: 编码后的embedding，形状为 [B, num_rerank, D] 其中 D = item_embedding_dim * 10
        """
        return self._forward_impl(model_input, encoded_embedding)

    @staticmethod
    def _clamp_fp16(tensor: torch.Tensor) -> torch.Tensor:
        """FP16下钳制NaN/Inf到安全范围，无NPU→CPU同步"""
        if tensor.dtype == torch.float16:
            return torch.clamp(
                torch.nan_to_num(tensor, nan=0.0, posinf=65504.0, neginf=-65504.0),
                min=-65504.0, max=65504.0)
        return tensor

    def _forward_impl(self, model_input: Dict[str, torch.Tensor], encoded_embedding: torch.Tensor = None):
        batch_size = model_input.get(self.cand_items_key).shape[0]
        num_rerank = model_input.get(self.cand_items_key).shape[1]
        if self.use_hstu:
            embedding_dim = encoded_embedding.shape[2]
            # 验证维度是否符合预期
            expected_dim = self.seq_input_dim
            if embedding_dim != expected_dim:
                logging.warning(f"Encoded embedding dimension mismatch. Expected: {expected_dim}, Got: {embedding_dim}")
            seq_embeddings = encoded_embedding

        dnn_feature_embs = []
        cross_feature_embs = []
        ppnet_feature_embs = []
        final_feature_embs = []

        cand_feature_embs = None
        need_cand_feature_embs = (
                self.use_dlrm_dnn
                or self.use_dlrm_cross
                or self.use_dlrm_ppnet
                or self.use_dlrm_linear
        )
        if need_cand_feature_embs:
            cand_feature_embs = self._get_feature_embeddings(model_input, self.candidate_dlrm_features)
            if cand_feature_embs.dim() == 2:
                cand_feature_embs = cand_feature_embs.view(batch_size, num_rerank, -1)

        if self.use_dlrm_dnn:
            dnn_feature_embs.append(cand_feature_embs)
        if self.use_dlrm_cross:
            cross_feature_embs.append(cand_feature_embs)
        if self.use_dlrm_ppnet:
            ppnet_feature_embs.append(cand_feature_embs)

        if self.use_hstu:
            if self.seq_as_cross_input and self.use_dlrm_cross:
                cross_feature_embs.append(seq_embeddings)
            if self.seq_as_dnn_input and self.use_dlrm_dnn:
                dnn_feature_embs.append(seq_embeddings)
            if self.seq_as_ppnet_input and self.use_dlrm_ppnet:
                ppnet_feature_embs.append(seq_embeddings)
            if self.seq_as_final_input:
                final_feature_embs.append(seq_embeddings)

        # CAN - 需要处理batch维度扩展
        if self.use_dlrm_can and self.cand_can_dim > 0:
            can_feature_embs = self._get_can_embeddings(model_input, self.can_selected_features)  # [bs, cand_size, D]
            if can_feature_embs.numel() > 0:  # 确保CAN特征不为空
                # 将CAN特征扩展到匹配seq_embeddings的维度
                if can_feature_embs.dim() == 2:  # [B, D_can]
                    can_feature_embs = can_feature_embs.view(batch_size, num_rerank, -1)

                if self.can_as_dnn_input and self.use_dlrm_dnn:
                    dnn_feature_embs.append(can_feature_embs)
                if self.can_as_dcn_input and self.use_dlrm_cross:
                    cross_feature_embs.append(can_feature_embs)
                if self.can_as_final_input:
                    final_feature_embs.append(can_feature_embs)

        # Cross - 添加维度检查和异常处理
        if self.use_dlrm_cross and cross_feature_embs:
            cross_feature_embs_concat = self._concat_features(cross_feature_embs, "cross_feature_emb")
            cross_feature_embs_out = self._clamp_fp16(self.cross(cross_feature_embs_concat))
            final_feature_embs.append(cross_feature_embs_out)

        # DNN - 添加维度检查和异常处理
        if self.use_dlrm_dnn and dnn_feature_embs:
            dnn_feature_embs_concat = self._concat_features(dnn_feature_embs, "dnn_feature_emb")
            if self.use_baseline_dcn:
                dnn_input_feature_embs = self.norm_before_dnn(dnn_feature_embs_concat)
                dnn_feature_embs_out = self.dnn(dnn_input_feature_embs)
                processed_dnn_output_embs = self._clamp_fp16(self.norm_after_dnn(dnn_feature_embs_out))
                final_feature_embs.append(processed_dnn_output_embs)
            else:
                dnn_feature_embs_out = self._clamp_fp16(self.dnn(dnn_feature_embs_concat))
                final_feature_embs.append(dnn_feature_embs_out)

        # PPNet - 添加维度检查和异常处理
        if self.use_dlrm_ppnet and ppnet_feature_embs:
            ppnet_feature_embs_concat = self._concat_features(ppnet_feature_embs, "ppnet_feature_emb")
            ppnet_gate_embs = self._get_feature_embeddings(model_input, self.ppnet_gate_features)
            # 扩展gate特征到匹配seq_embeddings的维度
            if ppnet_gate_embs.dim() == 2:  # [B, D_gate]
                ppnet_gate_embs = ppnet_gate_embs.view(batch_size, num_rerank, -1)

            ppnet_feature_embs_out = self._clamp_fp16(self.ppnet([ppnet_feature_embs_concat, ppnet_gate_embs]))
            final_feature_embs.append(ppnet_feature_embs_out)

        # 基线PPNet
        if self.use_baseline_ppnet:
            (concat_domain_embedding,
             concat_domain_no_target_embedding,
             domain_type1_embedding) = self._get_baseline_ppnet_embeddings(model_input)
            baseline_ppnet_embs_out = self._clamp_fp16(self.baseline_ppnet(concat_domain_embedding,
                                                                           concat_domain_no_target_embedding,
                                                                           domain_type1_embedding))
            final_feature_embs.append(baseline_ppnet_embs_out)

        # Linear - 添加维度检查和异常处理
        if self.use_dlrm_linear:
            linear_feature_embs = self._clamp_fp16(self.dlrm_linear(cand_feature_embs))
            final_feature_embs.append(linear_feature_embs)

        # 处理最终特征为空的情况
        if not final_feature_embs:
            if self.use_hstu:
                logging.info("No additional features from DLRM")
                # 如果没有任何特征被添加，返回seq_embeddings
                return seq_embeddings.view(batch_size, num_rerank, -1)
            else:
                logging.info("No feature embedding after DLRM, return all zero")
                return torch.zeros(batch_size, num_rerank, self.seq_input_dim)

        # Final concatenation - 添加维度检查
        final_feature_embs = self._clamp_fp16(torch.cat(final_feature_embs, dim=-1))
        return final_feature_embs, final_feature_embs.shape[-1]
