import abc
import logging
from typing import Dict, List

import torch
import torch.nn.init as init
import torch_npu

from modeling.generic.initialization import truncated_normal
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.transformers import GLUFFN, RMSNorm_npu
from modeling.generic.utils.constants import Const, FeatConst
from modeling.model_registry import ModelRegistry


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
class LocalEmbeddingModuleWithSideInfo(EmbeddingModule):
    """
    带有sideinfo的Embedding模块, 用于生成物品和用户的表示。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feature_conf = common_hp.get("feature_conf")
        model_conf = common_hp.get("model_conf")

        candidate_feature_columns: Dict = feature_conf.get('candidate_item_feature_columns', None)
        history_feature_columns: Dict = feature_conf.get('history_item_feature_columns', None)
        user_feature_columns: Dict = feature_conf.get('user_feature_columns', None)
        if user_feature_columns is None or candidate_feature_columns is None or history_feature_columns is None:
            raise ValueError(
                "user_feature_columns, candidate_feature_columns and history_feature_columns cannot be None")
        self.all_feature_names = list(candidate_feature_columns.keys()) \
                                 + list(history_feature_columns.keys()) \
                                 + list(user_feature_columns.keys())

        self.padding_index = feature_conf.get("padding_index", FeatConst.DFLT_PAD_IDX)
        self.multi_value_prefix = feature_conf.get('multi_value_prefix', FeatConst.DFLT_MULTI_VAL_PFX)
        self._item_embedding_dim = model_conf.get("item_embedding_dim", FeatConst.ITEM_EMB_DIM)

        self.feature_groups = feature_conf.get("feature_groups")
        if self.feature_groups is None:
            raise ValueError("feature_groups cannot be None")
        # action types
        self.action_types = feature_conf.get("action_types")
        # 是否使用gluffn
        self.use_gluffn = model_conf.get("use_gluffn", False)
        self.use_emb_rmsnorm = model_conf.get("use_emb_rmsnorm", False)
        self.amp_dtype = model_conf.get("amp_dtype", "fp16")
        self._debug_sanitize = model_conf.get("debug_sanitize", None)
        if self._debug_sanitize is None:
            # 默认：仅在AMP FP16启用时开启async（clamp防溢出）
            # use_amp=False时全fp32，无需sanitize；BF16范围足够也无需
            train_conf = common_hp.get("train_conf", {})
            _use_amp = train_conf.get("use_amp", False)
            self._debug_sanitize = "async" if (_use_amp and self.amp_dtype == "fp16") else False
        self.history_embedding_mode = model_conf.get("history_embedding_mode", "per_action")
        if self.history_embedding_mode not in {"per_action", "unified"}:
            raise ValueError("history_embedding_mode must be 'per_action' or 'unified'")
        logging.info("Embedding history_embedding_mode is %s", self.history_embedding_mode)

        required_keys = {"candidate", "history", "user"}
        if not required_keys.issubset(self.feature_groups.keys()):
            raise ValueError("candidate, history and user must in feature_groups to set mlps")

        self.emb_table = torch.nn.ModuleDict()
        self.dtypes = {}
        self.feature_conf = feature_conf
        self._initialize_embeddings()

        # sid_fusion_pos="input": 必须在 _init_mlps/_init_gluffns 之前，
        # 让 _init_mlps/_init_gluffns 在计算 input_dim 时加上 _sid_dim
        model_hp = model_cfg.get(Const.HP, {})
        self.sid_fusion_pos = model_hp.get("sid_fusion_pos", "model")
        self.use_sid = model_hp.get("use_sid", False)
        if self.sid_fusion_pos == "input" and self.use_sid:
            sid_D = model_hp.get("sid_D", feature_conf.get("sid_D", 64))
            sid_Ks = model_hp.get("sid_Ks", feature_conf.get("sid_K", [256, 256, 256]))
            self.sid_agg_type = model_hp.get("sid_agg_type", feature_conf.get("sid_agg_type", "concat"))
            sid_num_layers = len(sid_Ks)
            sid_emb_layers = []
            if len(sid_Ks) == 1:
                sid_Ks = sid_Ks * sid_num_layers
            for k in sid_Ks:
                sid_emb_layers.append(torch.nn.Embedding(k + 1, sid_D, padding_idx=0))
            self.sid_emb_layers = torch.nn.ModuleList(sid_emb_layers)
            # 加载 appid2codes
            num_items = feature_conf['candidate_item_feature_columns']['app_id']['feature_count']
            if model_hp.get("app_zero", False):
                num_items += 1
            sid_file_path = model_hp.get("appid2sids_path")
            appid2sids = torch.zeros(num_items, sid_num_layers, dtype=torch.int64)
            logging.info(f'Loading appid2sids from {sid_file_path}')
            import pandas as pd
            appid2sids_df = pd.read_csv(sid_file_path, sep='|', header=None, names=['AppId', 'SIDs'])
            for row in appid2sids_df.itertuples():
                app_id = row.AppId
                sids = torch.tensor(eval(row.SIDs), dtype=torch.int64)
                appid2sids[app_id] = sids
            logging.info(
                f'SIDs cover ratio: {(appid2sids.sum(dim=-1) != 0).sum() / len(appid2sids)} '
                f'(非零app_id数: {(appid2sids.sum(dim=-1) != 0).sum()}, 总app_id数: {len(appid2sids)})')
            logging.info(f'appid2sids[0:10]: {appid2sids[0:10].tolist()}')
            self.register_buffer("appid2codes", appid2sids, persistent=True)  # [N_item, num_codes]
            self.sid_num_code_layers = sid_num_layers
            self.sid_D = sid_D
            if self.sid_agg_type == "concat":
                self._sid_dim = sid_D * sid_num_layers
            else:
                self._sid_dim = sid_D
            logging.info("sid_fusion_pos=input: sid_emb_layers (%s layers, sid_D=%s, agg=%s, sid_dim=%s)",
                         sid_num_layers, sid_D, self.sid_agg_type, self._sid_dim)
        else:
            self._sid_dim = 0

        if self.use_gluffn:
            self._init_gluffns(self.feature_groups)
        else:
            self._init_mlps(self.feature_groups)
        if self.use_emb_rmsnorm:
            self._init_rmsnorms(self.feature_groups)

        self.use_time_fixed_token = model_conf.get("use_time_fixed_token", False)
        if self.use_time_fixed_token:
            self.cand_ts_key = feature_conf.get('candidate_timestamps_column', None)
            self.hist_ts_key = feature_conf.get('history_timestamps_column', None)
            feature_dim = feature_conf.get('common_feature_dim', 64)
            self.emb_table.update({
                'year': torch.nn.Embedding(52, feature_dim, padding_idx=self.padding_index),
                'month': torch.nn.Embedding(13, feature_dim, padding_idx=self.padding_index),
                'day': torch.nn.Embedding(32, feature_dim, padding_idx=self.padding_index),

                'doy': torch.nn.Embedding(367, feature_dim, padding_idx=self.padding_index),
                'weekday': torch.nn.Embedding(8, feature_dim, padding_idx=self.padding_index),
                'week': torch.nn.Embedding(54, feature_dim, padding_idx=self.padding_index),

                'hour': torch.nn.Embedding(25, feature_dim, padding_idx=self.padding_index),
                'time_bucket': torch.nn.Embedding(5, feature_dim, padding_idx=self.padding_index),
                'weekend': torch.nn.Embedding(3, feature_dim, padding_idx=self.padding_index),
            })
            self.time_combine_mlp = torch.nn.Linear(feature_dim * 9, self._item_embedding_dim)

        self.reset_params()

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim

    @staticmethod
    def debug_str() -> str:
        return f"LocalEmbeddingModuleWithSideInfo"

    @staticmethod
    def _get_base_feature_name(feature_name: str) -> str:
        feature_name = feature_name.replace(FeatConst.HIST_PFX, "")
        feature_name = feature_name.replace(FeatConst.CAND_PFX, "")

        return feature_name

    def _initialize_embeddings(self):
        feature_columns = ["candidate_item_feature_columns", "history_item_feature_columns", "user_feature_columns"]
        for conf_key in feature_columns:
            self._initialize_feature_embeddings(conf_key)

    def _initialize_feature_embeddings(self, feature_type: str) -> None:
        all_feature_name = self.feature_conf.get(feature_type, None)
        if all_feature_name is None:
            return

        for feature_name, feature_info in all_feature_name.items():
            base_feature_name = self._get_base_feature_name(feature_name)

            feature_count = feature_info.get('feature_count', FeatConst.FEAT_CNT)
            feature_dim = feature_info.get('dim', FeatConst.FEAT_DIM)
            feature_enabled = feature_info.get("enabled", True)
            feature_dtype = feature_info.get("dtype", FeatConst.DFLT_DTYPE)
            if feature_enabled:
                self.dtypes[feature_name] = feature_dtype
            if base_feature_name in self.emb_table:
                continue
            elif feature_enabled:
                if feature_dtype == "con":
                    # 连续特征：使用BatchNorm1d处理连续值
                    layer = torch.nn.BatchNorm1d(1)
                elif feature_dtype == "context":
                    layer = torch.nn.Embedding(feature_count + 1, feature_dim, padding_idx=self.padding_index)
                    init.uniform_(layer.weight, -0.05, 0.05)
                elif feature_dtype == "multi" or feature_dtype == "int":
                    shared_feat_name = feature_info.get("shared_feat_name", "")
                    if shared_feat_name not in self.all_feature_names:
                        raise ValueError("multifeat %s, shared_feat_name \"%s\" not found in feature columns",
                                         feature_name, shared_feat_name)
                    base_shared_feature_name = self._get_base_feature_name(shared_feat_name)
                    if base_shared_feature_name in self.emb_table:  # 存在，表示share的特征已创建，直接赋值
                        layer = self.emb_table[base_shared_feature_name]
                    else:  # 不存在，1. share的是自己，例如llm tag特征；2. share的特征尚未创建
                        feature_count = all_feature_name.get(shared_feat_name).get('feature_count', FeatConst.FEAT_CNT)
                        feature_dim = all_feature_name.get(shared_feat_name).get('dim', FeatConst.FEAT_DIM)
                        layer = torch.nn.Embedding(
                            feature_count + 1, feature_dim, padding_idx=self.padding_index)
                        init.uniform_(layer.weight, -0.05, 0.05)
                        self.emb_table[base_shared_feature_name] = layer
                else:
                    logging.error("feature_dtype %s is undefined for %s.", feature_dtype, feature_name)
                    continue
                self.emb_table[base_feature_name] = layer

    def _use_action_specific_history(self) -> bool:
        return self.history_embedding_mode == "per_action"

    def _get_history_feature_names(self, group_att: Dict, action_type: str = None) -> List[str]:
        if self._use_action_specific_history():
            if action_type is None:
                raise ValueError("action_type must be set when history_embedding_mode is 'per_action'")
            action_att = group_att.get(action_type)
            if action_att is None:
                raise ValueError(f"history action_type '{action_type}' not found in feature_groups")
            feature_names = action_att.get("features")
        else:
            feature_names = group_att.get("features")
            if feature_names is None and self.action_types and len(self.action_types) == 1:
                action_att = group_att.get(self.action_types[0], {})
                feature_names = action_att.get("features")
            if feature_names is None:
                nested_feature_groups = [
                    value.get("features")
                    for value in group_att.values()
                    if isinstance(value, dict) and value.get("features") is not None
                ]
                if len(nested_feature_groups) == 1:
                    feature_names = nested_feature_groups[0]

        if not feature_names:
            raise ValueError(
                "history feature_groups must define features for the selected history_embedding_mode"
            )
        return feature_names

    def _get_group_feature_names(self, group_name: str, group_att: Dict, action_type: str = None) -> List[str]:
        if group_name == FeatConst.HIST_PFX:
            return self._get_history_feature_names(group_att, action_type)

        feature_names = group_att.get("features")
        if not feature_names:
            raise ValueError(f"{group_name} feature_groups must define non-empty features")
        return feature_names

    def _init_mlps(self, feature_groups: Dict) -> None:
        for group_name, group_att in feature_groups.items():
            if group_name == "candidate_dlrm":
                continue
            fusion_type = group_att.get("fusion", "mlp")
            if fusion_type == "concat":
                continue
            sid_dim = self._sid_dim if (self.sid_fusion_pos == "input" and group_name != FeatConst.USER_PFX) else 0
            if group_name == FeatConst.HIST_PFX and self._use_action_specific_history():
                if not self.action_types:
                    raise ValueError("action_types must be set when history_embedding_mode is 'per_action'")
                for action in self.action_types:
                    feature_names = self._get_group_feature_names(group_name, group_att, action)
                    input_dim = self._calculate_input_dim(feature_names) + sid_dim
                    setattr(self, f"_{group_name}_{action}_mlp", self._create_mlp(input_dim, group_att.get("mlp")))
            else:
                feature_names = self._get_group_feature_names(group_name, group_att)
                input_dim = self._calculate_input_dim(feature_names) + sid_dim
                setattr(self, f"_{group_name}_mlp", self._create_mlp(input_dim, group_att.get("mlp")))

    def _init_gluffns(self, feature_groups: Dict) -> None:
        for group_name, group_att in feature_groups.items():
            if group_name == "candidate_dlrm":
                continue
            fusion_type = group_att.get("fusion", "gluffn")
            if fusion_type == "concat":
                continue
            sid_dim = self._sid_dim if (self.sid_fusion_pos == "input" and group_name != FeatConst.USER_PFX) else 0
            if group_name == FeatConst.HIST_PFX and self._use_action_specific_history():
                if not self.action_types:
                    raise ValueError("action_types must be set when history_embedding_mode is 'per_action'")
                for action in self.action_types:
                    feature_names = self._get_group_feature_names(group_name, group_att, action)
                    input_dim = self._calculate_input_dim(feature_names) + sid_dim
                    setattr(self, f"_{group_name}_{action}_gluffn",
                            self._create_gluffn(input_dim, group_att.get("gluffn")))
            else:
                feature_names = self._get_group_feature_names(group_name, group_att)
                input_dim = self._calculate_input_dim(feature_names) + sid_dim
                setattr(self, f"_{group_name}_gluffn", self._create_gluffn(input_dim, group_att.get("gluffn")))

    def _init_rmsnorms(self, feature_groups: Dict) -> None:
        for group_name, group_att in feature_groups.items():
            if group_name == "candidate_dlrm":
                continue
            sid_dim = self._sid_dim if (self.sid_fusion_pos == "input" and group_name != FeatConst.USER_PFX) else 0
            if group_name == FeatConst.HIST_PFX and self._use_action_specific_history():
                if not self.action_types:
                    raise ValueError("action_types must be set when history_embedding_mode is 'per_action'")
                for action in self.action_types:
                    feature_names = self._get_group_feature_names(group_name, group_att, action)
                    input_dim = self._calculate_input_dim(feature_names) + sid_dim
                    setattr(self, f"_{group_name}_{action}_rmsnorm", RMSNorm_npu(input_dim, eps=1e-6))
            else:
                feature_names = self._get_group_feature_names(group_name, group_att)
                input_dim = self._calculate_input_dim(feature_names) + sid_dim
                setattr(self, f"_{group_name}_rmsnorm", RMSNorm_npu(input_dim, eps=1e-6))

    def _calculate_input_dim(self, feature_names: List) -> int:
        input_dim = 0
        for feat_name in feature_names:
            if self.dtypes.get(feat_name) == "con":
                input_dim += 1
            else:
                base_feat_name = self._get_base_feature_name(feat_name)
                input_dim += self.emb_table[base_feat_name].weight.shape[1]

        return input_dim

    @staticmethod
    def _has_non_finite(value: torch.Tensor) -> bool:
        return not torch.isfinite(value).all()

    @staticmethod
    def _log_non_finite(prefix: str, value: torch.Tensor) -> None:
        finite = torch.isfinite(value)
        if finite.all():
            return
        nan_count = torch.isnan(value).sum()
        inf_mask = torch.isinf(value)
        posinf_count = (inf_mask & (value > 0)).sum()
        neginf_count = (inf_mask & (value < 0)).sum()
        logging.info(
            "%s has non-finite values: nan=%s, posinf=%s, neginf=%s, shape=%s, dtype=%s",
            prefix,
            nan_count,
            posinf_count,
            neginf_count,
            tuple(value.shape),
            value.dtype,
        )

    def _sanitize_tensor(self, prefix: str, value: torch.Tensor) -> torch.Tensor:
        if not self._debug_sanitize:
            return value
        # NPU→CPU同步版本：检测+日志+替换，调试用
        if self._debug_sanitize == "verbose":
            if self._has_non_finite(value):
                self._log_non_finite(prefix, value)
                return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return value
        # 无同步版本：用clamp钳制到安全范围，NaN通过nan_to_num原地替换
        # 整个计算留在NPU上，不触发任何CPU同步
        if value.dtype == torch.float16:
            value = torch.nan_to_num(value, nan=0.0, posinf=65504.0, neginf=-65504.0)
            value = torch.clamp(value, min=-65504.0, max=65504.0)
        return value

    def _sanitize_time_tensor(self, prefix: str, value: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(value):
            value = self._sanitize_tensor(prefix, value)
        return value

    def _sanitize_embedding_weight(self, feature_name: str, layer: torch.nn.Module) -> None:
        if not self._debug_sanitize:
            return
        weight = getattr(layer, "weight", None)
        if weight is None:
            return
        # NPU→CPU同步版本：检测+日志+替换，调试用
        if self._debug_sanitize == "verbose":
            if self._has_non_finite(weight):
                self._log_non_finite(f"{feature_name} embedding weight", weight)
                with torch.no_grad():
                    weight.copy_(torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0))
            return
        # 无同步版本：用clamp钳制，不触发CPU同步
        if weight.dtype == torch.float16:
            with torch.no_grad():
                weight.copy_(torch.clamp(
                    torch.nan_to_num(weight, nan=0.0, posinf=65504.0, neginf=-65504.0),
                    min=-65504.0, max=65504.0))

    def _create_mlp(self, input_dim: int, mlp_att: Dict) -> torch.nn.Sequential:
        available_act = {
            "relu": torch.nn.ReLU,
            "gelu": torch.nn.GELU,
            "tanh": torch.nn.Tanh,
            "sigmoid": torch.nn.Sigmoid,
            "silu": torch.nn.SiLU
        }

        hidden_dims = mlp_att.get("hidden_dims")
        act_fn = mlp_att.get("act_fn")
        act_cls = available_act.get(act_fn.lower() if isinstance(act_fn, str) else act_fn, None)
        if act_cls is None:
            logging.info("Set activation function in Embedding MLP to None")

        layers = []

        cur_dim = input_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(cur_dim, h))
            if act_cls:
                layers.append(act_cls())
            cur_dim = h

        layers.append(torch.nn.Linear(cur_dim, self._item_embedding_dim))

        return torch.nn.Sequential(*layers)

    def _create_gluffn(self, input_dim: int, gluffn_att: Dict) -> GLUFFN:
        if gluffn_att is None:
            logging.warning('gluffn_att is None!!!')
            ffn_dim_multiplier = None
        else:
            ffn_dim_multiplier = gluffn_att.get("ffn_dim_multiplier", None)
        return GLUFFN(input_dim, input_dim, self._item_embedding_dim, ffn_dim_multiplier=ffn_dim_multiplier)

    def reset_params(self):
        seen = set()
        for name, params in self.named_parameters():
            if 'emb' in name:
                ptr = params.data_ptr()
                if ptr in seen:
                    continue  # 已初始化过底层 tensor
                seen.add(ptr)
                truncated_normal(params, mean=0.0, std=0.02)
            elif 'weight' in name and 'mlp' in name:
                torch.nn.init.normal_(params, mean=0.0, std=0.01)
            elif 'bias' in name and 'mlp' in name:
                torch.nn.init.constant_(params, 0.0)
            elif 'weight' in name and 'rmsnorm' in name:
                torch.nn.init.ones_(params)
            else:
                logging.info("Skipping initializing params %s - not configured", name)
        with torch.no_grad():
            for layer in self.emb_table.values():
                if isinstance(layer, torch.nn.Embedding) and layer.padding_idx is not None:
                    layer.weight[layer.padding_idx].zero_()

    def _get_feature_embeddings(
            self,
            features: Dict[str, torch.Tensor],
            feature_names: List[str],
            action_type: str = None
    ) -> torch.Tensor:
        feature_emb_list = []

        # 先正常处理所有特征
        for _, feature_name in enumerate(feature_names):
            feature_id = features[feature_name]
            feature_dtype = self.dtypes.get(feature_name)
            base_feature_name = self._get_base_feature_name(feature_name)
            layer = self.emb_table[base_feature_name]

            if feature_dtype == "con":
                # 处理连续特征（保持原逻辑不变）
                feature_id = self._sanitize_tensor(f"{feature_name} input", feature_id)
                if feature_id.dim() == 1:
                    feature_id = feature_id.unsqueeze(-1)
                    feature_id_for_bn = feature_id.unsqueeze(-1)
                elif feature_id.dim() == 2:
                    feature_id_for_bn = feature_id.unsqueeze(1)
                else:
                    feature_id_for_bn = feature_id.transpose(1, 2)
                try:
                    feature_value = layer(feature_id_for_bn)
                    feature_value = feature_value.transpose(1, 2)
                except Exception as e:
                    logging.info("feature_id_for_bn shape is %s, feature_id_for_bn dim is %s", feature_id_for_bn.shape,
                                 feature_id_for_bn.dim())
                    logging.error("Failed to compute embedding for %s. Exception: %s", feature_name, e)
                    raise e
            else:
                # 处理离散特征（保持原逻辑不变）
                try:
                    self._sanitize_embedding_weight(feature_name, layer)
                    feature_value = layer(feature_id)
                except Exception as e:
                    logging.error(f"Error processing feature: {feature_name}")
                    logging.error(f"Base feature name: {base_feature_name}")
                    logging.error(f"Feature ID shape: {feature_id.shape}")
                    logging.error(f"Feature ID dtype: {feature_id.dtype}")
                    raise e
                if feature_value.ndim == 4:
                    # 多值特征才会是4维(历史序列embedding之后就是三维的) -> 只有candidate和user侧会有
                    # 先除再sum：每个元素先除以valid_count再求和，部分和≈mean量级，FP16也不会溢出
                    mask = (feature_id != self.padding_index).unsqueeze(-1).to(feature_value.dtype)
                    valid_count = mask.sum(dim=2, keepdim=True)
                    feature_value = (feature_value * mask / (valid_count + 1)).sum(dim=2)

            feature_value = self._sanitize_tensor(f"{feature_name} embedding output", feature_value)
            feature_emb_list.append(feature_value)

        # 拼接特征嵌入
        feature_embs = torch.cat(feature_emb_list, dim=-1)
        feature_embs = self._sanitize_tensor("feature_embs before mlp", feature_embs)
        return feature_embs

    # ========== SID 辅助方法（sid_fusion_pos="input" 时使用） ==========

    @torch.no_grad()
    def _map_app_to_codes(self, appid_mapped: torch.Tensor) -> torch.Tensor:
        """app_id -> [c1, c2, ..., cL]，返回 (..., L) long tensor。"""
        flat = appid_mapped.view(-1)
        num_appid = self.appid2codes.size(0)
        invalid_appid = (flat < 0) | (flat >= num_appid)
        flat_safe = flat.masked_fill(invalid_appid, 0).long()
        codes_flat = self.appid2codes[flat_safe]
        out_shape = (*appid_mapped.shape, self.sid_num_code_layers)
        codes = codes_flat.view(out_shape)
        return codes

    def _lookup_sid_raw_emb(self, appid: torch.Tensor) -> torch.Tensor:
        """
        查 SID code embedding，按 sid_agg_type 聚合。
        返回: (B, N or M, sid_dim)
        sid_dim = sid_D * num_code_layers (concat) 或 sid_D (pool)。
        """
        sids = self._map_app_to_codes(appid)  # (B, N or M, L)
        emb_list = []
        for lvl, emb in enumerate(self.sid_emb_layers):
            ids_lvl = sids[..., lvl]
            emb_lvl = emb(ids_lvl)  # (B, N or M, sid_D)
            emb_list.append(emb_lvl)  # [(B, N, 64), ...]

        stacked = torch.stack(emb_list, dim=-2)  # (B, N or M, L, sid_D)
        if self.sid_agg_type == "pool":
            sid_emb = stacked.mean(dim=-2)  # (B, N or M, sid_D)
        else:  # "concat"
            sid_emb = torch.cat(emb_list, dim=-1)  # (B, N or M, sid_D * L)
        return sid_emb

    def get_group_embedding(self, group_name: str, input_features: Dict[str, torch.Tensor],
                            action_type: str = None) -> torch.Tensor:
        group_dict = self.feature_groups.get(group_name)
        fusion_type = group_dict.get("fusion", "gluffn" if self.use_gluffn else "mlp")

        if group_name == FeatConst.HIST_PFX and self._use_action_specific_history() and action_type is not None:
            # 获得对应行为的dict
            feature_names = self._get_group_feature_names(group_name, group_dict, action_type)
            action_history_embs = self._get_feature_embeddings(
                input_features,
                feature_names,
                action_type
            )
            # sid_fusion_pos="input": 将 SID raw embedding concat 到 feature_embs 末尾
            if self.sid_fusion_pos == "input" and self._sid_dim > 0 and group_name != FeatConst.USER_PFX:
                appid = input_features.get(self.feature_conf.get("history_items_key"))
                if appid is not None:
                    sid_emb = self._lookup_sid_raw_emb(appid)
                    action_history_embs = torch.cat([action_history_embs, sid_emb], dim=-1)

            if self.use_emb_rmsnorm:
                rmsnorm = getattr(self, f"_{group_name}_{action_type}_rmsnorm")
                action_history_embs = rmsnorm(action_history_embs)

            if fusion_type == "gluffn":
                feature_embs = getattr(self, f"_{group_name}_{action_type}_gluffn")(action_history_embs)
            elif fusion_type == "mlp":
                feature_embs = getattr(self, f"_{group_name}_{action_type}_mlp")(action_history_embs)
            else:
                # fusion_type == "concat": 直接拼接，不做线性映射
                feature_embs = action_history_embs
            feature_embs = self._sanitize_tensor(f"{group_name}_{action_type} feature_embs after mlp", feature_embs)
        else:
            feature_embs = self._get_feature_embeddings(
                input_features,
                self._get_group_feature_names(group_name, group_dict)
            )
            # sid_fusion_pos="input": 将 SID raw embedding concat 到 feature_embs 末尾
            if self.sid_fusion_pos == "input" and self._sid_dim > 0 and group_name != FeatConst.USER_PFX:
                if group_name == FeatConst.CAND_PFX:
                    items_key = self.feature_conf.get("candidate_items_key")
                else:
                    items_key = self.feature_conf.get("history_items_key")
                appid = input_features.get(items_key)
                if appid is not None:
                    sid_emb = self._lookup_sid_raw_emb(appid)
                    feature_embs = torch.cat([feature_embs, sid_emb], dim=-1)

            if self.use_emb_rmsnorm:
                rmsnorm = getattr(self, f"_{group_name}_rmsnorm")
                feature_embs = rmsnorm(feature_embs)

            if fusion_type == "gluffn":
                feature_embs = getattr(self, f"_{group_name}_gluffn")(feature_embs)
            elif fusion_type == "mlp":
                feature_embs = getattr(self, f"_{group_name}_mlp")(feature_embs)
            elif fusion_type == "concat":
                pass
            else:
                raise ValueError(f"Unknown fusion type: {fusion_type}")

            feature_embs = self._sanitize_tensor(f"{group_name} feature_embs after mlp", feature_embs)

        # 对candidate token附加time_fixed_token
        if self.use_time_fixed_token and group_name != FeatConst.USER_PFX:
            ts_tensor = None
            if group_name == FeatConst.CAND_PFX:
                ts_tensor = input_features[self.cand_ts_key]  # [B, S]
            elif group_name == FeatConst.HIST_PFX:
                ts_tensor = input_features[self.hist_ts_key]  # [B, S]
            if ts_tensor is None:
                return feature_embs
            ts_tensor = self._sanitize_time_tensor(f"{group_name} timestamp input", ts_tensor)
            batch_size, seq_len = ts_tensor.shape

            valid_mask = (ts_tensor > 0) & (ts_tensor < 2147483647)
            ts_int = torch.where(valid_mask, ts_tensor, torch.zeros_like(ts_tensor)).long()

            # 1. unix seconds decomposition
            days = torch.div(ts_int, 86400, rounding_mode='floor')
            secs_of_day = ts_int % 86400
            hour = torch.div(secs_of_day, 3600, rounding_mode='floor')

            # coarse time segment
            # 0: midnight [0,6)
            # 1: morning [6,12)
            # 2: afternoon [12,18)
            # 3: evening [18,24)
            time_bucket = torch.div(hour, 6, rounding_mode='floor')

            # 2. Gregorian calendar parsing
            z = days + 719468
            era = torch.div(torch.where(z >= 0, z, z - 146096), 146097, rounding_mode='floor')
            doe = z - era * 146097

            yoe = torch.div(
                doe - torch.div(doe, 1460, rounding_mode='floor')
                + torch.div(doe, 36524, rounding_mode='floor')
                - torch.div(doe, 146096, rounding_mode='floor'),
                365,
                rounding_mode='floor'
            )

            y = yoe + era * 400
            doy = doe - (365 * yoe + torch.div(yoe, 4, rounding_mode='floor') - torch.div(yoe, 100,
                                                                                          rounding_mode='floor'))

            mp = torch.div(5 * doy + 2, 153, rounding_mode='floor')
            day = doy - torch.div(153 * mp + 2, 5, rounding_mode='floor') + 1

            month = mp + torch.where(mp < 10, 3, -9)
            year = y + (month <= 2).long()

            # 3. derived calendar features
            # weekday: 1~7 (Mon~Sun)
            weekday = ((days + 3) % 7) + 1

            week_of_year = torch.div(doy, 7, rounding_mode='floor') + 1

            # weekend flag
            is_weekend = (weekday >= 6).long()

            # 4. embedding indices
            # reserve 0 for missing
            year_idx = torch.where(valid_mask, (year - 1999).clamp(1, 51), torch.zeros_like(year)).view(-1)
            month_idx = torch.where(valid_mask, month.clamp(1, 12), torch.zeros_like(month)).view(-1)
            day_idx = torch.where(valid_mask, day.clamp(1, 31), torch.zeros_like(day)).view(-1)

            doy_idx = torch.where(valid_mask, (doy + 1).clamp(1, 366), torch.zeros_like(doy)).view(-1)
            weekday_idx = torch.where(valid_mask, weekday.clamp(1, 7), torch.zeros_like(weekday)).view(-1)
            week_idx = torch.where(valid_mask, week_of_year.clamp(1, 53), torch.zeros_like(week_of_year)).view(-1)

            hour_idx = torch.where(valid_mask, (hour + 1).clamp(1, 24), torch.zeros_like(hour)).view(-1)
            time_bucket_idx = torch.where(valid_mask, (time_bucket + 1).clamp(1, 4),
                                          torch.zeros_like(time_bucket)).view(-1)

            weekend_idx = torch.where(valid_mask, is_weekend + 1, torch.zeros_like(is_weekend)).view(-1)

            # 5. embedding lookup
            year_emb = self.emb_table['year'](year_idx)
            month_emb = self.emb_table['month'](month_idx)
            day_emb = self.emb_table['day'](day_idx)

            doy_emb = self.emb_table['doy'](doy_idx)
            weekday_emb = self.emb_table['weekday'](weekday_idx)
            week_emb = self.emb_table['week'](week_idx)

            hour_emb = self.emb_table['hour'](hour_idx)
            time_bucket_emb = self.emb_table['time_bucket'](time_bucket_idx)
            weekend_emb = self.emb_table['weekend'](weekend_idx)

            # 6. combine
            time_features = torch.cat([
                year_emb, month_emb, day_emb,
                doy_emb, weekday_emb, week_emb,
                hour_emb, time_bucket_emb, weekend_emb
            ], dim=-1)

            time_features = self._sanitize_tensor(f"{group_name} time_features before mlp", time_features)
            time_token = self.time_combine_mlp(time_features).view(batch_size, seq_len, -1)
            time_token = self._sanitize_tensor(f"{group_name} time_token", time_token)
            time_token = time_token * valid_mask.unsqueeze(-1).to(time_token.dtype)

            feature_embs = feature_embs + time_token
            feature_embs = self._sanitize_tensor(f"{group_name} feature_embs after time token", feature_embs)

        return feature_embs

    def get_ui_embeddings(self, group_name: str, input_features: Dict[str, torch.Tensor],
                          action_type: str = None) -> torch.Tensor:
        """
            根据特征组名获取embedding。

            :param group_name: 特征组名: "candidate" 或者 "history" 或者 "user"
            :param input_features: 特征字典
            :param llm_emb_dict: llm_embedding字典
            :param action_type: 可选变量，行为类别
            :return: 物品嵌入张量。
        """
        if group_name == FeatConst.CAND_PFX:  # 暂时设定其他group全部为candidate侧
            feature_group = list(self.feature_groups.keys())
            feature_group.remove(FeatConst.HIST_PFX)
            feature_group.remove(FeatConst.USER_PFX)
        else:
            feature_group = [group_name]

        if group_name == FeatConst.HIST_PFX and self._use_action_specific_history():
            # 历史特征必须有action_type
            if action_type is None:
                raise ValueError("action_type must not be None when history_embedding_mode is 'per_action'")
            else:
                emb_list = []
                for group_name in feature_group:
                    if group_name == "candidate_dlrm":
                        continue
                    emb_list.append(self.get_group_embedding(group_name, input_features, action_type))
        else:
            emb_list = []
            for group_name in feature_group:
                if group_name == "candidate_dlrm":
                    continue
                emb_list.append(self.get_group_embedding(group_name, input_features))

        # 多组embedding求和在FP16下可能溢出，upcast到FP32求和后再转回
        stacked = torch.stack(emb_list, dim=0)
        if stacked.dtype == torch.float16:
            final_emb = stacked.float().sum(dim=0).to(stacked.dtype)
        else:
            final_emb = stacked.sum(dim=0)

        return final_emb


@ModelRegistry.register()
class UnifiedLocalEmbeddingModuleWithSideInfo(LocalEmbeddingModuleWithSideInfo):
    """
    Backward-compatible name for unified history embedding.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        common_hp = dict(common_hp)
        model_conf = dict(common_hp.get("model_conf", {}))
        model_conf.setdefault("history_embedding_mode", "unified")
        common_hp["model_conf"] = model_conf
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

    @staticmethod
    def debug_str() -> str:
        return "UnifiedLocalEmbeddingModuleWithSideInfo"
