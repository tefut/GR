from typing import Tuple

import torch

DEFAULT_DATA_LEN = 62568324
L2NORM_STRING = "l2"
LAYERNORM_STRING = "ln"
MAX_K = 2500


class Const:
    """
    定义配置文件以及框架中使用的常量
    """
    BEST_LOSS = 1e+7
    EPS = 1e-4
    TransformerCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    LABEL_DICT = []
    MODULE_NAME = "name"
    IS_CUSTOMIZE = "is_customize"
    PATH = "path"
    CLS_NAME = "cls_name"
    HP = "hp"
    SUB_MODELS = "sub_models"
    MODEL_CFG = "model_cfg"
    COMMON_HP = "common_hp"
    EXPECTED_NUM_UNIQUE_ITEMS = 250000


class FeatConst:
    """
    定义特征里常用的常量
    """
    DFLT_PAD_IDX = 0
    DFLT_MULTI_VAL_PFX = "pref_"
    ITEM_EMB_DIM = 128
    FEAT_CNT = 10
    FEAT_DIM = 32
    DFLT_DTYPE = "int"
    HIST_PFX = "history"
    CAND_PFX = "candidate"
    USER_PFX = "user"
    DFLT_N_TOKEN_PER_ITEM = 2
    DFLT_HIST_TS_KEY = "history_timestamps"
    DFLT_CAND_TS_KEY = "candidate_timestamps"
    DFLT_HIST_RATINGS_KEY = "history_action_type"
    DFLT_CAND_RATINGS_KEY = "candidate_action_type"
    DFLT_HIST_ITEM_KEY = "history_item_id"
    DFLT_CAND_ITEM_KEY = "candidate_item_id"
    DFLT_HIST_DATE_KEY = "history_date"
    DFLT_CAND_DATE_KEY = "candidate_date"
