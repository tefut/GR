import os
from dataclasses import dataclass
from typing import List

import torch

from data.concat_dataset import DatasetMusicV2LONGER
from modeling.generic.utils.constants import Const


@dataclass
class RecoDataset:
    """
    数据集类，用于存储训练和评估数据集的相关信息。
    """
    max_sequence_length: int
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset


def get_format_csv(data_dir, pth) -> str:
    return os.path.join(data_dir, pth)


def get_reco_dataset(
        dataset: str,
        data_dir: str,
        max_sequence_length: int,
        rank: int,
        world_size: int,
        pth: str,
        chronological: bool = True,
        feature_conf: dict = None,
        num_rerank=256,
        cut_off_time: int = None
) -> RecoDataset:
    """
    创建并返回一个RecoDataset对象, 包含训练和评估数据集。

    :param data_dir: 数据目录。
    :param max_sequence_length: 序列的最大长度。
    :param rank: 全局rank ID。
    :param world_size: 分布式并行总进程数。
    :param pth: 文件路径。
    :param chronological: 是否按时间正序排列。
    :param feature_conf: 特征配置字典。
    :return: RecoDataset对象, 包含训练和评估数据集的相关信息。
    """
    if dataset == "music-scalingraw-longer":
        config_dir_path = feature_conf.get("config_folder_name", "config")
        dataset_config_dir = os.path.join(data_dir, config_dir_path)
        data_dir_path = feature_conf.get("data_folder_name", "data")
        train_dir_path = feature_conf.get("train_folder_name", "train_orc")
        valid_dir_path = feature_conf.get("valid_folder_name", "valid_orc")
        dataset_train_dir = os.path.join(data_dir, data_dir_path, train_dir_path)
        dataset_valid_dir = os.path.join(data_dir, data_dir_path, valid_dir_path)
        seq_feat_config = feature_conf.get('seq_feature_columns')
        train_dataset = DatasetMusicV2LONGER(
            dataset_config_dir=dataset_config_dir,
            user_feat_config=feature_conf.get('user_feature_columns'),
            item_feat_config=feature_conf.get('item_feature_columns'),
            seq_feat_config=seq_feat_config,
            dataset_train_dir=dataset_train_dir,
            dataset_valid_dir=dataset_valid_dir,
            rank=rank,
            world_size=world_size,
            is_train=True,
            file_format="orc",
            exclude_features=feature_conf.get('exclude_features', [])
        )
        eval_dataset = DatasetMusicV2LONGER(
            dataset_config_dir=dataset_config_dir,
            user_feat_config=feature_conf.get('user_feature_columns'),
            item_feat_config=feature_conf.get('item_feature_columns'),
            seq_feat_config=feature_conf.get('seq_feature_columns'),
            dataset_train_dir=dataset_train_dir,
            dataset_valid_dir=dataset_valid_dir,
            rank=rank,
            world_size=world_size,
            is_train=False,
            file_format="orc",
            exclude_features=feature_conf.get('exclude_features', []),
            num_rerank=num_rerank
        )

    return RecoDataset(
        max_sequence_length=max_sequence_length,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
