from math import dist
import os
import sys
import time

import json
import yaml

from datetime import datetime, timedelta
from typing import Dict
from modeling.generic.utils.constants import Const

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


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
    if dataset == "Longer_dataset_ulan" or dataset == "Game":
        if os.path.exists(feature_map_dir_or_path):
            feature_map_file = read_json(feature_map_dir_or_path)
            feature_map = feature_map_file['maxIndexMap']
            gauc_querys = model_conf.get('gauc_querys', {})
            logging.info("gauc_querys: %s", gauc_querys)
            _feature_conf = [feature_conf["candidate_item_feature_columns"],
                             feature_conf["history_item_feature_columns"], feature_conf["user_feature_columns"]]
            for conf in _feature_conf:
                for col_name, col_info in conf.items():
                    if col_name in feature_map:
                        col_info['feature_count'] = feature_map[col_name]
            logging.info("Updated feature_count from featuremap file: %s", feature_map_dir_or_path)

        period = feature_conf.get("period")
        date_format = "%Y%m%d-%H%M%S"
        date_obj = datetime.strptime(period, date_format)
        cut_off_date_obj = date_obj - timedelta(days=int(feature_conf["cut_off_bias"]))
        cut_off_date_obj = cut_off_date_obj.replace(hour=0, minute=0, second=0, microsecond=0)

        logging.info("cut off time is set to %s", cut_off_date_obj.strftime(date_format))

        cut_off_date = int(cut_off_date_obj.strftime('%Y%m%d'))
        feature_conf["cut_off_time"] = cut_off_date
    else:
        raise NotImplementedError("Unknown dataset")

    # 当配置使用GRModelEp模型时(embedding parallel，当前使用torchrec实现)时，检查依赖的算子是否安装，否则fallback回GR_model
    if model_cfg[Const.MODULE_NAME] == "GRModelEp":
        from modeling.generic.sequential import HAS_TORCHREC
        if not HAS_TORCHREC:
            logging.warning("Required ops of torchrec not installed, fallback model to 'GR_model'")
            model_cfg[Const.MODULE_NAME] = "GR_model"
    model_conf["root_model_type"] = model_cfg[Const.MODULE_NAME]
    return feature_conf, model_conf, feature_map, gauc_querys
