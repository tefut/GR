from typing import Tuple

import torch

L2NORM_STRING = "l2"
LAYERNORM_STRING = "ln"


class Const:
    """
    定义配置文件以及框架中使用的常量
    """
    BEST_LOSS = 1e+7
    EPS = 1e-7
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
