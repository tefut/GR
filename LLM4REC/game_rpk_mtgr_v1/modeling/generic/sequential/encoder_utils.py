from __future__ import annotations

import importlib

from modeling.generic.utils import _model_register
from modeling.generic.utils.constants import Const

_MODULES = [
    "modeling.generic.sequential.embedding_modules",
    "modeling.generic.sequential.input_features_preprocessors",
    "modeling.generic.sequential.output_postprocessors",
    "modeling.generic.sequential.attn_mask_modules",
    "modeling.generic.sequential.prediction_modules",
    "modeling.generic.sequential.loss_mask_modules",
    "modeling.generic.sequential.loss_modules",
    "modeling.generic.sequential.negative_sampler",
    "modeling.generic.sequential.rab_modules",
    "modeling.generic.sequential.transformers",
    "modeling.generic.sequential.GR_model",
]


def load_modules():
    for module in _MODULES:
        importlib.import_module(module)


def get_sequential_encoder_v2(model_init_config):
    """
    递归加载所有模块

    :param model_init_config: 完整的自定义配置文件

    :return gr_model: 根据所选的模块和配置生成的初始化的GR_model实例
    """

    load_modules()
    model_cls_dict = _model_register.MODEL_CLS_DICT
    root_model_type = model_init_config[Const.MODEL_CFG][Const.MODULE_NAME]
    gr_model = model_cls_dict.get(root_model_type)(
        model_init_config[Const.MODEL_CFG], model_init_config[Const.COMMON_HP], model_cls_dict)

    return gr_model