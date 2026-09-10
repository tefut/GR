from __future__ import annotations
import logging
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
import torch
from modeling.generic.sequential.attn_mask_modules import AttentionMaskModule
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.dlrm import DLRM
from modeling.generic.sequential.embedding_modules import EmbeddingModule
from modeling.generic.sequential.input_features_preprocessors import InputFeaturesPreprocessorModule
from modeling.generic.sequential.loss_modules import LossModule
from modeling.generic.sequential.negative_sampler import NegativesSampler
from modeling.generic.sequential.output_postprocessors import OutputPostprocessorModule
from modeling.generic.sequential.prediction_modules import FeedForwardModule
from modeling.generic.sequential.score_boost import apply_recent_paid_download_boost
from modeling.generic.sequential.srn_module import SRNModule
from modeling.generic.sequential.transformers import SequentialModule, TransformerCacheState
from modeling.generic.utils.constants import Const, FeatConst
from modeling.model_registry import ModelRegistry
@ModelRegistry.register(
    req_subs={"EmbeddingModule", "InputFeaturesPreprocessorModule", "SequentialModule", "AttentionMaskModule",
              "FeedForwardModule", "OutputPostprocessorModule", "LossModule", "SIM"}, opt_subs={"NegativesSampler"})
class ActCondMTGRModel(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        # get configs
        self._verbose = model_cfg[Const.HP].get("verbose", True)
        model_conf = common_hp["model_conf"]
        feat_conf = common_hp["feature_conf"]
        train_conf = common_hp["train_conf"]
        export_conf = common_hp["export_conf"]
        self._phase = train_conf.get("phase", "pretrain")
        self._load_sim_embedding = train_conf.get("load_sim_embedding", True)
        self._load_ano_sim = train_conf.get("load_ano_sim", False)
        self.history_length = common_hp['data_loader_conf'].get('history_length', 400)
        self.use_dynamic_padding = common_hp['data_loader_conf'].get('use_dynamic_padding', False)
        self.batch_size = train_conf.get("local_batch_size", 512)
        self.item_embedding_dim = model_conf.get("item_embedding_dim", 256)
        if model_conf.get('use_user_embeddings_for_rerank', False):
            self.item_embedding_dim *= 2
        # 是否用HSTU
        self.use_hstu = model_conf.get("use_hstu", True)
        # 是否使用SRN
        self.use_enhanced_interest_embeddings = model_conf.get("use_enhanced_interest_embeddings", False)
        # 是否用dlrm
        self.use_dlrm = model_conf.get("use_dlrm", False)
        self.use_baseline_dcn = model_conf.get('use_baseline_dcn', False)
        self.use_baseline_ppnet = model_conf.get('use_baseline_ppnet', False)
        # 时间戳特征调整开关
        self.truncate_timestamps_feature = model_conf.get("truncate_timestamps_feature", False)
        self.pt_d_truncate_times = model_conf.get("pt_d_truncate_times", 1e6)
        self.oper_time_truncate_times = model_conf.get("oper_time_truncate_times", 1e3)
        self.use_heat_weight = model_conf.get("use_heat_weight", False)
        self.recent_paid_download_boost_conf = model_conf.get("recent_paid_download_time_boost", {})
        # register buffer
        self.register_buffer("history_lengths", torch.tensor(self.history_length, dtype=torch.int))
        self.num_rerank = common_hp['data_loader_conf'].get('num_rerank', 400)
        self.feature_groups = feat_conf.get("feature_groups", None)
        # history keys
        self.history_feature_column_names = list(feat_conf.get("history_item_feature_columns", {}).keys())
        self.hist_date_key = feat_conf.get("history_date_column", FeatConst.DFLT_HIST_DATE_KEY)
        self.hist_ts_key = feat_conf.get("history_timestamps_column", FeatConst.DFLT_HIST_TS_KEY)
        self.hist_items_key = feat_conf.get("history_items_key", FeatConst.DFLT_HIST_ITEM_KEY)
        # action types
        self.action_types = feat_conf["action_types"]
        self.action_mapping = {action: i + 1 for i, action in enumerate(self.action_types)}
        # candidate keys
        self.cand_date_key = feat_conf.get("candidate_date_column", FeatConst.DFLT_CAND_DATE_KEY)
        self.cand_ts_key = feat_conf.get("candidate_timestamps_column", FeatConst.DFLT_CAND_TS_KEY)
        self.cand_items_key = feat_conf.get("candidate_items_key", FeatConst.DFLT_CAND_ITEM_KEY)
        self.cand_ratings_key = feat_conf.get("candidate_ratings_column", FeatConst.DFLT_CAND_RATINGS_KEY)
        # embedding
        model_conf.setdefault("history_embedding_mode", "unified")
        self.embedding_module: EmbeddingModule = self.init_sub_model("EmbeddingModule")
        # 注册SRN Module
        if self.use_enhanced_interest_embeddings:
            self.srn_module: SRNModule = self.init_sub_model("SRNModule")
        # 预训练相关
        self._frozen_embedding = train_conf.get("frozen_embedding", False)
        if self._phase == "train" and self._load_sim_embedding: