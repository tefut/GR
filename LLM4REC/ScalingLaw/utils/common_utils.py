import json
from typing import Dict

import yaml

from modeling.generic.utils.constants import Const
from utils.logging_utils import logging


def read_yml(file_path, use_e=False):
    """
    :param file_path: 文件路径
    :param use_e: yml文件中是否含有科学计数法
    :return:
    """
    with open(file_path, 'r', encoding='UTF-8') as yml_file:
        result = yaml.safe_load(yml_file)
    return result


def read_json(json_path):
    """
    读取json格式文件
    :param json_path: 文件路径
    :return:
    """
    with open(json_path, "r", encoding='utf-8') as f:
        result = json.loads(f.read())
    return result


def get_config(file_path):
    """
    :param file_path: 文件路径
    :return config_dict: 配置文件字典
    """
    # 读取配置文件
    config_dict = read_yml(file_path, use_e=True)

    return config_dict


def weird_division(x, y):
    """
    :param x: 被除数
    :param y: 除数
    :return x/y: 商
    """
    if y < Const.EPS and y >= 0:
        y = Const.EPS
    elif y > -Const.EPS and y < 0:
        y = -Const.EPS
    return x / y


def refine_feat_and_model_conf(dataset: str,
                               feature_conf: Dict,
                               model_conf: Dict,
                               feature_map_dir_or_path,
                               model_cfg, **kwargs):
    """
    1. 设置cut_off_time
    2. 从featuremap中读取每个特征的最大值, 写入feature_conf中每个用户/商品特征的feature_count字段.
    3. 根据feature_conf里positive_behavior字段指定的正向行为，配置行为到标签的映射和行为总数。

    """

    if dataset == "music-scalingraw-longer":
        feature_map = read_json(feature_map_dir_or_path)
        _feature_conf = [feature_conf["item_feature_columns"], feature_conf["user_feature_columns"]]
        for _, seq_config in feature_conf["seq_feature_columns"].items():
            _feature_conf.append(seq_config)
        for f in _feature_conf:
            for col_name, col_info in f.items():
                col_info['feature_count'] = feature_map['maxIndexMap'].get(col_name, 0)
                logging.info(
                    "feature %s have %s identical values "
                    "(it is normal if you see 0 when the feature is a continous feature)",
                    col_name, col_info['feature_count'])

        feature_map_out = feature_map['sparse']
        

    else:
        raise NotImplementedError("Unknown dataset")

    # 当配置使用GRModelEp模型时(embedding parallel，当前使用torchrec实现)时，检查依赖的算子是否安装，否则fallback回GR_model
    if model_cfg[Const.MODULE_NAME] == "LongerEp":
        enable_fusion_ops = model_cfg[Const.SUB_MODELS]["SequentialModule"][
            Const.SUB_MODELS]["Transformer"][Const.HP].get("enable_fusion_ops", False)
        from modeling import HAS_TORCHREC
        if not HAS_TORCHREC or not enable_fusion_ops:
            logging.warning("Required ops of torchrec not installed or enabled, fallback model to 'LONGER'")
            model_cfg[Const.MODULE_NAME] = "LONGER"
            model_cfg[Const.CLS_NAME] = "LONGER"
    model_conf["root_model_type"] = model_cfg[Const.MODULE_NAME]
    return feature_conf, model_conf, feature_map_out


def compute_user_item_feature_dims(feat_conf, model_conf):
    return feat_conf["user_emb_dims"], feat_conf["item_emb_dims"]
