from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from torch.autograd.profiler import record_function

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
class MultiSeqMTGRModel(BaseModel):
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
        self.features_need_repeat = model_conf.get("features_need_repeat", [])

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
        # profile_time: 开启各阶段耗时统计，关闭时零额外开销
        self._profile_time = model_conf.get("profile_time", False)
        self.recent_paid_download_boost_conf = model_conf.get("recent_paid_download_time_boost", {})

        self.spl_recov_flag = model_conf.get("val_spl_recov", False)

        # register buffer
        self.register_buffer("history_lengths", torch.tensor(self.history_length, dtype=torch.int))

        self.num_rerank = common_hp['data_loader_conf'].get('num_rerank', 400)
        self.feature_groups = feat_conf.get("feature_groups", None)

        # history keys
        self.history_feature_column_names = list(feat_conf.get("history_item_feature_columns", {}).keys())
        self.hist_date_key = feat_conf.get("history_date_column", FeatConst.DFLT_HIST_DATE_KEY)
        self.hist_ts_key = feat_conf.get("history_timestamps_column", FeatConst.DFLT_HIST_TS_KEY)
        self.hist_items_key = feat_conf.get("history_items_key", FeatConst.DFLT_HIST_ITEM_KEY)
        self.hist_ratings_key = feat_conf.get("history_ratings_column", FeatConst.DFLT_HIST_RATINGS_KEY)

        self.linear_weigh = model_conf.get("linear_weigh", 1000)
        # action types
        self.action_types = feat_conf["action_types"]
        self.action_mapping = {action: i for i, action in enumerate(self.action_types)}

        # candidate keys
        self.cand_date_key = feat_conf.get("candidate_date_column", FeatConst.DFLT_CAND_DATE_KEY)
        self.cand_ts_key = feat_conf.get("candidate_timestamps_column", FeatConst.DFLT_CAND_TS_KEY)
        self.cand_items_key = feat_conf.get("candidate_items_key", FeatConst.DFLT_CAND_ITEM_KEY)
        self.cand_ratings_key = feat_conf.get("candidate_ratings_column", FeatConst.DFLT_CAND_RATINGS_KEY)

        # embedding (延迟到 SID 参数注入之后)
        if self.use_enhanced_interest_embeddings:
            self.srn_module: SRNModule = self.init_sub_model("SRNModule")

        # 预训练相关
        self._frozen_embedding = train_conf.get("frozen_embedding", False)
        if self._phase == "train" and self._load_sim_embedding:
            if not self._load_ano_sim:
                save_dir = train_conf['save_dir']
                checkpoint_dir = os.path.join(save_dir, export_conf["save_dir_name"])
                embedding_weights_path = f"{checkpoint_dir}/embedding_module.pth"
                pretrained_state_dict = torch.load(embedding_weights_path, map_location=torch.device('cpu'))
                self.load_embedding_dict(pretrained_state_dict)
                logging.info("load sim emb")
                if self._frozen_embedding:
                    self.freeze_embeddings()
                    logging.info("freeze sim emb!")
            elif os.path.isfile(train_conf['load_ano_sim_path']):
                embedding_weights_path = train_conf['load_ano_sim_path']
                pretrained_state_dict = torch.load(embedding_weights_path, map_location=torch.device('cpu'))
                self.load_embedding_dict(pretrained_state_dict)
                logging.info("load ano sim emb")
                if self._frozen_embedding:
                    self.freeze_embeddings()
                    logging.info("freeze sim emb!")
            else:
                logging.warning("No sim emb!!!")

        model_hp = model_cfg[Const.HP]

        # 将 SID 参数注入到子模块的 model_cfg 中
        # （子模块是 sub_model，看不到父级 hp，但 sid_fusion_pos="input" 时需要 SID 参数）
        _sub_cfgs = self.model_cfg.get(Const.SUB_MODELS, {})
        for _sub_key in ("InputFeaturesPreprocessorModule", "EmbeddingModule"):
            _sub_cfg = _sub_cfgs.get(_sub_key)
            if _sub_cfg is not None:
                _sub_hp = _sub_cfg.setdefault(Const.HP, {})
                _sub_hp.setdefault("use_sid", model_hp.get("use_sid", False))
                _sub_hp.setdefault("sid_fusion_pos", model_hp.get("sid_fusion_pos", "model"))
                _sub_hp.setdefault("sid_D", model_hp.get("sid_D", feat_conf.get("sid_D", 64)))
                _sub_hp.setdefault("sid_Ks", model_hp.get("sid_Ks", feat_conf.get("sid_K", [256, 256, 256])))
                _sub_hp.setdefault("sid_agg_type", model_hp.get("sid_agg_type",
                                                                feat_conf.get("sid_agg_type", "concat")))
                _sub_hp.setdefault("app_zero", model_hp.get("app_zero", False))
                _sub_hp.setdefault("appid2sids_path", model_hp.get("appid2sids_path", ""))

        self.embedding_module: EmbeddingModule = self.init_sub_model("EmbeddingModule")

        # sequence modeling
        self.input_propcessor_module: InputFeaturesPreprocessorModule = self.init_sub_model(
            "InputFeaturesPreprocessorModule")
        if self._phase == "train" or self._phase == "test_case":
            self.sequence_model: SequentialModule = self.init_sub_model("SequentialModule")
        else:
            self.sim_model = self.init_sub_model("SIM")

        # negative sampler
        self.negative_sampler: NegativesSampler = None if "NegativesSampler" not in model_cfg[Const.SUB_MODELS] \
            else self.init_sub_model("NegativesSampler")
        if self.negative_sampler is not None:
            self.negative_sampler.load_embedding_module(self.embedding_module)
        self._max_sequence_length: int = self.history_length + self.num_rerank

        # use user embeddings
        self.concat_user_embeddings = model_conf.get("use_user_embeddings_for_rerank", False)

        fuse_ia = feat_conf.get("fuse_ia", False)
        self.token_per_item = 2 if not fuse_ia else 1

        self.attention_mask_module: AttentionMaskModule = self.init_sub_model("AttentionMaskModule")

        # 预测头
        if self.use_dlrm:
            # # DLRM应与HSTU共用embedding_module
            model_cfg['sub_models']['DLRMModule']['embedding_module'] = self.embedding_module
            model_cfg['sub_models']['DLRMModule']['item_embedding_dim'] = model_conf['item_embedding_dim']
            self.dlrm_module: DLRM = self.init_sub_model("DLRMModule")
            feedfoward_module = self.model_cfg['sub_models']['FeedForwardModule']['sub_models']
            # 此处更新feedforward_module的config内的input_dim,修改参数无用
            if 'LinearModuleForRerankScore' in feedfoward_module:
                feedfoward_module['LinearModuleForRerankScore']['hp']['input_dim'] = (
                    self.dlrm_module.final_input_dim)
            if 'FeedForwardModuleForRerankScore' in feedfoward_module:
                feedfoward_module['FeedForwardModuleForRerankScore']['hp']['input_dim'] = (
                    self.dlrm_module.final_input_dim)
            if 'FeedForwardModuleForNextActionPred' in feedfoward_module:
                feedfoward_module['FeedForwardModuleForNextActionPred']['hp']['input_dim'] = (
                    self.dlrm_module.final_input_dim)
        self.feed_forward_module: FeedForwardModule = self.init_sub_model("FeedForwardModule")

        self.output_processor_module: OutputPostprocessorModule = self.init_sub_model("OutputPostprocessorModule")
        self.loss_module: LossModule = self.init_sub_model("LossModule")

        self.use_sid = model_hp.get("use_sid", False)
        self.sid_fusion_pos = model_hp.get("sid_fusion_pos", "model")  # "model" 或 "input"
        if self.use_sid:
            if self.sid_fusion_pos == "model":
                self.sid_emb_type = feat_conf.get("sid_emb_type", "simple")
                self.sid_Ks = feat_conf.get("sid_K", [256, 256, 256])
                self.num_code_layers = len(self.sid_Ks)
                self.sid_H = feat_conf.get("sid_H", 1e6)
                self.sid_prefix_n = feat_conf.get("sid_prefix_n", 3)
                self.get_sid(model_hp, feat_conf)
                (self.sid_emb_layers,
                 self.hist_sid_mlp,
                 self.cand_sid_mlp,
                 self.cand_shared_feats) = self.create_sid_embedding_layers(feat_conf, model_conf, self.num_code_layers)
                self.use_sid_gate = feat_conf.get("use_sid_gate", False)
                if self.use_sid_gate:
                    logging.info("use sid gate")
                    item_emb_dim = model_conf.get("item_embedding_dim")
                    self.hist_sid_gate = self.create_gate_layers(item_emb_dim)
                    self.cand_sid_gate = self.create_gate_layers(item_emb_dim)
        self.reset_params()

    def freeze_embeddings(self):
        """
        冻结 embedding_module 中所有 embedding 层的参数，使其不参与训练。
        """
        logging.info("开始冻结所有 embedding 参数...")
        emb_table = self.embedding_module.emb_table

        # 遍历 embedding 表中的每一个 embedding 层
        for feature_name, emb_layer in emb_table.items():
            # 将该层的参数设置为不需要计算梯度
            emb_layer.weight.requires_grad = False
            logging.info(f"  -> 已冻结 feature '{feature_name}' 的 embedding 参数。")

        logging.info("所有 embedding 参数已成功冻结。")

    def unfreeze_embeddings(self):
        """
        解冻所有 embedding 参数，使其可以参与训练。
        """
        logging.info("开始解冻所有 embedding 参数...")
        emb_table = self.embedding_module.emb_table

        # 遍历 embedding 表中的每一个 embedding 层
        for feature_name, emb_layer in emb_table.items():
            # 将该层的参数设置为需要计算梯度
            emb_layer.weight.requires_grad = True
            logging.info(f"  -> 已解冻 feature '{feature_name}' 的 embedding 参数。")

        logging.info("所有 embedding 参数已成功解冻。")

    def load_embedding_dict(self, old_state_dict: dict):
        """
        将旧的 embedding state_dict 加载到新的模型中，处理 num_embeddings 增大的情况。

        Args:
            old_state_dict (dict): 包含预训练权重的 state_dict。
        """
        new_embeddings_module_dict = self.embedding_module.emb_table

        with torch.no_grad():
            for feature_name, new_emb_layer in new_embeddings_module_dict.items():
                logging.info(f"feature_name is: {feature_name}")
                old_weight_key = f"emb_table.{feature_name}.weight"

                if old_weight_key in old_state_dict:
                    old_weight_tensor = old_state_dict[old_weight_key]

                    old_size, old_dim = old_weight_tensor.shape

                    new_size = new_emb_layer.num_embeddings
                    new_dim = new_emb_layer.embedding_dim

                    if new_dim != old_dim:
                        logging.info("Feature '%s' 的 embedding_dim 不匹配 (%s vs %s)。跳过加载。",
                                     feature_name, new_dim, old_dim)
                        continue

                    logging.info("正在加载 Feature '%s': 旧大小=%s, 新大小=%s",
                                 feature_name, old_size, new_size)

                    if new_size >= old_size:
                        # 将旧的权重张量复制到新 embedding 层权重的前面部分
                        new_emb_layer.weight.data[:old_size, :] = old_weight_tensor
                        logging.info("成功加载 %s 条记录。新增的 %s 条记录保留随机初始化。",
                                     old_size, new_size - old_size)
                    else:
                        # 新表更小，只加载能装下的部分
                        new_emb_layer.weight.data = old_weight_tensor[:new_size, :]
                        logging.info("成功加载 %s 条记录（截断）。", new_size)
                else:
                    logging.info("Feature '%s' 在旧权重中未找到 (尝试的键: '%s')，保留其随机初始化。",
                                 feature_name, old_weight_key)

    def reset_params(self):
        for name, params in self.named_parameters():
            if ("sequence_model" in name) or ("embedding_module" in name):
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
            except ValueError:
                if self._verbose:
                    logging.info("Failed to initialize %s: %s params", name, params.data.shape[0])

    @staticmethod
    def create_gate_layers(item_emb_dim):
        """
        输入：
            item_emb_dim: int。物品 Embedding 的维度大小。
        输出：
            sid_gate: torch.nn.Sequential。
        """
        sid_gate = torch.nn.Sequential(
            torch.nn.Linear(2 * item_emb_dim, item_emb_dim),
            torch.nn.Sigmoid()
        )

        torch.nn.init.constant_(sid_gate[0].bias, -2.0)
        torch.nn.init.xavier_uniform_(sid_gate[0].weight)

        return sid_gate

    def create_sid_embedding_layers(self, feat_conf, model_conf, num_code_layers):
        """
        输入：
            feat_conf: 配置字典，包含 sid_D（每个 Code 的 Embedding 维度，默认 64）。
            model_conf: 配置字典，包含 item_embedding_dim（最终映射到的物品特征维度）。
            num_code_layers: int。SID 编码的层数。
        输出：
            emb_layers: torch.nn.ModuleList。包含多个 Embedding 层，每个层对应 SID 的一个层级。
            hist_mlp_layer: torch.nn.Linear。将历史序列中拼接后的 SID 向量映射到 item_emb_dim。
            cand_mlp_layer: torch.nn.Linear。将候选物品拼接后的 SID 向量映射到 item_emb_dim。
        """
        D = feat_conf.get("sid_D", 64)
        item_emb_dim = model_conf.get("item_embedding_dim")
        raw_sid_dim = D * self.num_code_layers
        emb_layers = []
        if self.sid_emb_type == "simple":
            if len(self.sid_Ks) == 1:
                self.sid_Ks = self.sid_Ks * num_code_layers
            for k in self.sid_Ks:
                emb_layers.append(torch.nn.Embedding(k + 1, D, padding_idx=0))
            emb_layers = torch.nn.ModuleList(emb_layers)

        hist_mlp_layer = torch.nn.Linear(raw_sid_dim, item_emb_dim)
        logging.info("history sid mlp in_dim %s, out_dim %s", raw_sid_dim, item_emb_dim)
        shared_feats = set()
        cand_sid_dim = raw_sid_dim
        cand_mlp_layer = torch.nn.Linear(cand_sid_dim, item_emb_dim)
        logging.info("candidate sid mlp in_dim %s, out_dim %s", raw_sid_dim, item_emb_dim)
        return emb_layers, hist_mlp_layer, cand_mlp_layer, shared_feats

    def get_sid(self, model_hp, feat_conf):
        # 获取sid
        num_items = feat_conf['candidate_item_feature_columns'][self.cand_items_key]['feature_count']
        if model_hp.get("app_zero", False):
            num_items += 1
        sid_file_path = model_hp.get("appid2sids_path")
        sid_file_type = sid_file_path.split('.')[-1]
        appid2sids = torch.zeros(num_items, len(self.sid_Ks), dtype=torch.int64)
        logging.info(f'Loading appid2sids from {sid_file_path}')
        logging.info(f'sid_file_type: {sid_file_type}')
        appid2sids_df = pd.read_csv(sid_file_path, sep='|', header=None, names=['AppId', 'SIDs'])
        for row in appid2sids_df.itertuples():
            app_id = row.AppId
            sids = torch.tensor(eval(row.SIDs), dtype=torch.int64)
            appid2sids[app_id] = sids  # appid [sid1, sid2, sid3]
        logging.info(f'SIDs cover ratio: {(appid2sids.sum(dim=-1) != 0).sum() / len(appid2sids)}')
        logging.info(f'appid2sids[1000:1100]: {appid2sids[1000: 1010]}')
        self.register_buffer("appid2codes", appid2sids, persistent=True)  # [N_item, num_codes]

    @torch.no_grad()
    def map_app_to_codes(self, appid_mapped: torch.Tensor):
        """
        输入:
            appid_mapped: 任意 shape 的 long Tensor（历史或候选的 appid_mapped）
        输出:
            codes: 同 shape 的 [..., L] long Tensor，每个位置是 [c1,c2,c3]
        """

        # flatten -> LUT 映射 sid_id -> 再映射到 codes_table
        flat = appid_mapped.view(-1)
        num_appid = self.appid2codes.size(0)
        invalid_appid = (flat < 0) | (flat >= num_appid)
        flat_safe = flat.masked_fill(invalid_appid, 0).long()
        codes_flat = self.appid2codes[flat_safe]
        # reshape 回原始 batch 形状
        out_shape = (*appid_mapped.shape, self.num_code_layers)  # [..., L]
        codes = codes_flat.view(out_shape)
        return codes, (~invalid_appid).view(appid_mapped.shape)

    def _lookup_sid_raw_emb(self, input_feat):
        """
        底层通用查表逻辑：从 input_feat (ID) -> raw_sid_embedding
        """
        sids, valid_items = self.map_app_to_codes(input_feat)
        if self.sid_emb_type == "simple":
            emb_list = []
            for lvl, emb in enumerate(self.sid_emb_layers):
                ids_lvl = sids[..., lvl]
                emb_lvl = emb(ids_lvl)
                emb_list.append(emb_lvl)
            out_emb = torch.cat(emb_list, dim=-1)
        else:
            raise ValueError(f"Unknown sid_emb_type: {self.sid_emb_type}")

        return out_emb

    def get_cand_sid_emb(self, model_inputs):
        target_feat = model_inputs.get(self.cand_items_key)
        target_sid = self._lookup_sid_raw_emb(target_feat)
        return self.cand_sid_mlp(target_sid)

    def get_hist_sid_emb(self, model_inputs):
        input_feat = model_inputs.get(self.hist_items_key)
        raw_emb = self._lookup_sid_raw_emb(input_feat)
        return self.hist_sid_mlp(raw_emb)

    def get_embeddings(self, model_inputs):
        candidate_embeddings = self.embedding_module.get_ui_embeddings(
            group_name=FeatConst.CAND_PFX, input_features=model_inputs
        )
        user_feature_embs = self.embedding_module.get_ui_embeddings(
            group_name=FeatConst.USER_PFX, input_features=model_inputs
        )
        if self.use_sid:
            if self.sid_fusion_pos == "model":
                cand_sid_embs = self.get_cand_sid_emb(model_inputs)
                if self.use_sid_gate:
                    g2 = self.cand_sid_gate(torch.cat([candidate_embeddings, cand_sid_embs], dim=-1))
                    candidate_embeddings = candidate_embeddings + g2 * cand_sid_embs
                else:
                    candidate_embeddings = candidate_embeddings + cand_sid_embs
        return user_feature_embs, candidate_embeddings

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
        综合序列信息，生成用户 embedding.
        [B, N] -> [B, N, D].
        """
        if x_offsets is None:
            # 修改后 - 避免不必要的设备迁移
            x_offsets = torch.cat((
                torch.zeros(1, dtype=past_lengths.dtype, device=past_lengths.device),
                torch.cumsum(past_lengths, dim=0)
            ), dim=0)

        item_embeddings, _ = self.sequence_model(
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
        # 如果推理时，只返回候选集的部分的输出的商品token；如果训练时，返回所有商品的输出的token。
        if torch.onnx.is_in_onnx_export() or num_rerank > 0:
            _item_embeddings = item_embeddings[:, -num_rerank:, :]
        else:
            _item_embeddings = item_embeddings[:, -num_rerank:, :]

        if self.concat_user_embeddings:
            user_embeddings = item_embeddings[:, :1, :].repeat(1, _item_embeddings.shape[1], 1)
            item_embeddings = torch.cat([user_embeddings, _item_embeddings], dim=-1)
        else:
            item_embeddings = _item_embeddings

        return self.output_processor_module(item_embeddings)

    def process_single_act_seq(self,
                               action: str,
                               num_rerank: int,
                               candidate_embeddings,
                               user_feature_embs,
                               model_input: dict,
                               precomputed_max_seq_len: int = None
                               ) -> torch.Tensor:
        """
        处理单个行为序列
        :param action: 行为类型
        :param num_rerank: 候选数量
        :param candidate_embeddings: 候选嵌入
        :param user_feature_embs: 用户嵌入
        :param model_input: 模型输入
        :param precomputed_max_seq_len: 预计算的_max_seq, 避免重复.item() NPU→CPU同步
        :return:
        """
        single_act_seq_input = {}
        # 获取行为对应的时间戳
        act_ts_key = self.hist_ts_key
        act_ts = model_input.get(act_ts_key)

        batch_size = candidate_embeddings.shape[0]

        for feat_name in self.feature_groups[FeatConst.HIST_PFX][action]['features']:
            single_act_seq_input[feat_name] = model_input[feat_name]
        for feat_name in self.feature_groups[FeatConst.CAND_PFX]['features']:
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
            else self._max_sequence_length)
        attn_mask = self.attention_mask_module(single_act_seq_input, _batch_max_seq_len, num_rerank)

        with record_function("## EmbeddingLookup ##"):
            if self._profile_time:
                _t0 = time.time()
            past_embeddings = self.embedding_module.get_ui_embeddings(
                group_name=FeatConst.HIST_PFX,
                input_features=single_act_seq_input,
                action_type=action
            )  # [batch_size, hist_length, emb_dim]

            if self.use_sid and self.sid_fusion_pos == "model":
                hist_sid_embs = self.get_hist_sid_emb(model_input)
                if self.use_sid_gate:
                    g1 = self.hist_sid_gate(torch.cat([past_embeddings, hist_sid_embs], dim=-1))
                    past_embeddings = past_embeddings + g1 * hist_sid_embs
                else:
                    past_embeddings = past_embeddings + hist_sid_embs

            # 使用SRN结构交叉用户历史序列和item特征
            if self.use_enhanced_interest_embeddings:
                candidate_emb_for_srn = candidate_embeddings  # [batch_size, num_rerank, emb_dim]

                max_seq_len = past_embeddings.shape[1]
                batch_size = past_embeddings.shape[0]

                if self.use_dynamic_padding:
                    _hist_lengths = model_input.get("history_lengths", self.history_lengths)
                    if _hist_lengths.dim() == 0:
                        _hist_lengths = _hist_lengths.unsqueeze(0).expand(batch_size)
                    sequence_mask = torch.arange(max_seq_len, device=past_embeddings.device)[None,
                                    :] < _hist_lengths.unsqueeze(1)
                else:
                    sequence_mask = torch.arange(max_seq_len, device=past_embeddings.device)[None,
                                    :] < self.history_lengths
                    sequence_mask = sequence_mask.expand(batch_size, -1)  # 扩展到批次大小

                try:
                    enhanced_past_embeddings = self.srn_module(
                        [candidate_emb_for_srn, past_embeddings],
                        mask=sequence_mask
                    )
                    # 原始历史序列应与SRN序列进行融合
                    past_embeddings = past_embeddings + enhanced_past_embeddings
                except Exception as e:
                    logging.info("SRN module error: %s", e)
                    raise e
            if self._profile_time:
                logging.info("[PROFILE] EmbeddingLookup+SRN+SID cost: %.3f s", time.time() - _t0)

        with record_function("## InputPreprocessor ##"):
            if self._profile_time:
                _t0 = time.time()
            past_lengths_after_input_processor, user_embeddings, _ = self.input_propcessor_module(
                history_embeddings=past_embeddings,
                candidate_embeddings=candidate_embeddings,
                history_lengths=model_input.get(
                    "history_lengths",
                    self.history_lengths) if self.use_dynamic_padding else self.history_lengths.expand(
                    batch_size),
                history_ids=model_input.get(self.hist_items_key),
                candidate_ids=model_input.get(self.cand_items_key),
                user_feature_embs=user_feature_embs,
                history_ratings=self.action_mapping.get(action),
                candidate_ratings=model_input.get(self.cand_ratings_key),
                hist_times=act_ts,
            )
            if self._profile_time:
                logging.info("[PROFILE] InputPreprocessor cost: %.3f s", time.time() - _t0)

        # 每个行为序列用自己的date_diff_seq
        with record_function("## Sequential ##"):
            if self._profile_time:
                _t0 = time.time()
            encoded_embeddings = self.generate_user_embeddings(past_lengths=past_lengths_after_input_processor,
                                                               all_timestamps=act_ts,
                                                               seq_embeddings=user_embeddings,
                                                               attn_mask=attn_mask,
                                                               num_rerank=num_rerank)
            if self._profile_time:
                logging.info("[PROFILE] Sequential cost: %.3f s", time.time() - _t0)

        return encoded_embeddings

    def process_multi_action_history_seq(self,
                                         num_rerank: int,
                                         candidate_embeddings,
                                         user_feature_embs,
                                         model_input: dict,
                                         precomputed_max_seq_len: int = None
                                         ) -> torch.Tensor:
        """
        综合处理所有行为序列
        :param num_rerank: 候选数量
        :param candidate_embeddings: 候选嵌入
        :param user_feature_embs: 用户嵌入
        :param model_input: 模型输入
        :param precomputed_max_seq_len: 预计算的_max_seq, 避免重复.item() NPU→CPU同步
        :return:
        """
        multi_action_history_seq_list = []
        action_types = self.action_types

        for action in action_types:
            multi_action_history_seq_list.append(
                self.process_single_act_seq(
                    action,
                    num_rerank,
                    candidate_embeddings,
                    user_feature_embs,
                    model_input,
                    precomputed_max_seq_len=precomputed_max_seq_len))
        # 每个行为序列的emb都是[bs, 1, item_embedding_dim * 2] -> 开启了use_user_embeddings_for_rerank
        # 不开启的话是[bs, 1, item_embedding_dim]
        encoded_embeddings = torch.cat(multi_action_history_seq_list, dim=-1)

        return encoded_embeddings

    def forward(
            self,
            model_input: dict,
            is_train: bool = True
    ) -> torch.Tensor | dict:
        """
        生成式推荐大模型前向传播过程

        :param model_input: 传入的字典，里面包括音乐的特征，用户的特征和其他序列信息。
        :param is_train: 是否是训练模式。
        """

        if torch.onnx.is_in_onnx_export():
            # 部分特征推理时需要进行repeat操作
            if self.features_need_repeat:
                for feat in self.features_need_repeat:
                    feat_value = model_input[feat]
                    model_input[feat] = feat_value[:, :1, ...].expand_as(feat_value)
            # 游戏/小游戏线上推理时时间需要截断
            if self.truncate_timestamps_feature:
                model_input[self.cand_date_key] = (
                        model_input[self.cand_date_key] // self.pt_d_truncate_times).to(torch.int64)
                model_input[self.cand_ts_key] = (
                        model_input[self.cand_ts_key] // self.oper_time_truncate_times).to(torch.int64)

        num_rerank = model_input.get(self.cand_items_key).shape[1]

        # 预计算_max_seq一次，避免在process_single_act_seq中每个行为类型重复.item() NPU→CPU同步
        if self.use_dynamic_padding:
            _hist_lens = model_input.get("history_lengths", self.history_lengths)
            _max_seq = (_hist_lens.max().cpu().item() + num_rerank) \
                if _hist_lens.dim() > 0 else self._max_sequence_length
        else:
            _max_seq = self._max_sequence_length

        if self._profile_time:
            _t0 = time.time()

        if self.use_hstu:
            # 多行为序列建模
            with record_function("## Embedding ##"):
                user_feature_embs, candidate_embeddings = self.get_embeddings(model_input)
            with record_function("## Sequential ##"):
                encoded_embeddings = self.process_multi_action_history_seq(num_rerank,
                                                                           candidate_embeddings,
                                                                           user_feature_embs,
                                                                           model_input,
                                                                           precomputed_max_seq_len=_max_seq)
        else:
            encoded_embeddings = torch.zeros(self.batch_size, num_rerank, self.item_embedding_dim)

        if self._profile_time:
            _t1 = time.time()
            logging.info("[PROFILE] Embedding+Sequential cost: %.3f s", _t1 - _t0)

        if self.use_dlrm:
            with record_function("## DLRM ##"):
                _t0 = time.time() if self._profile_time else None
                encoded_embeddings, final_feature_emb_dim = self.dlrm_module(model_input, encoded_embeddings)
                if self._profile_time:
                    logging.info("[PROFILE] DLRM cost: %.3f s", time.time() - _t0)

        if not is_train or torch.onnx.is_in_onnx_export():
            results = self.feed_forward_module(
                encoded_embeddings,
                self.spl_recov_flag or torch.onnx.is_in_onnx_export())
            rerank_score = results["rerank_score"]
            if self.use_heat_weight:
                rerank_score = apply_recent_paid_download_boost(
                    rerank_score,
                    model_input,
                    self.recent_paid_download_boost_conf,
                    self.cand_ts_key
                )
            results = {"rerank_score": rerank_score * self.linear_weigh}
            return results
        else:
            results = self.feed_forward_module(encoded_embeddings, return_logits=True)

            loss = self.loss_module(past_embeddings=None,
                                    encoded_embeddings=encoded_embeddings,
                                    predictions=results,
                                    model_inputs=model_input,
                                    negative_sampler=self.negative_sampler)
            return loss
