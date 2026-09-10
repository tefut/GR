import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import pandas as pd

from data.concat_dataset_v1 import DatasetMusicLonger, DatasetAG
from modeling.generic.utils.constants import Const


@dataclass
class RecoDataset:
    """
    数据集类，用于存储训练和评估数据集的相关信息。
    """
    max_sequence_length: int
    num_unique_items: int
    max_item_id: int
    all_item_ids: List[int]
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset
    use_dynamic_padding: bool = False


def get_format_csv(data_dir, pth) -> str:
    return os.path.join(data_dir, pth)


def get_reco_dataset(
        dataset: str,
        data_dir: str,
        train_data_path: str,
        valid_data_path: str,
        max_sequence_length: int,
        rank: int,
        world_size: int,
        chronological: bool = True,
        feature_conf: dict = None,
        num_rerank=256,
        history_length=400,
        use_dynamic_padding=False,
        padding_side_origin="left",
        padding_side="right"
) -> RecoDataset:
    """
    创建并返回一个RecoDataset对象, 包含训练和评估数据集。

    :param dataset: 数据集名称。
    :param data_dir: 数据目录。
    :param train_data_path: 训练数据集路径。
    :param valid_data_path: 验证数据集路径。
    :param max_sequence_length: 序列的最大长度。
    :param rank: 全局rank ID。
    :param world_size: 分布式并行总进程数。
    :param chronological: 是否按时间正序排列。
    :param feature_conf: 特征配置字典。
    :param num_rerank: 候选物品个数。
    :param history_length: 历史行为序列长度。
    :return: RecoDataset对象, 包含训练和评估数据集的相关信息。
    """
    if dataset == "Longer_dataset_ulan":
        token_per_item = 1 if feature_conf.get("fuse_ia", True) else 2
        cut_off_time = feature_conf.get("cut_off_bias", None)
        if cut_off_time is None:
            raise ValueError("cut_off_time in config should be set as date for music.")

        # 提取共同参数,避免重复代码
        dataset_params = {
            'ignore_last_n': 0,
            'chronological': chronological,
            'is_ads': feature_conf.get('time_desc', True),
            'sep': ';',
            'rank': rank,
            'world_size': world_size,
            'history_feature_columns': feature_conf.get('history_item_feature_columns'),
            'candidate_feature_columns': feature_conf.get('candidate_item_feature_columns'),
            'user_feature_columns': feature_conf.get('user_feature_columns'),
            'candidate_items_key': feature_conf.get('candidate_items_key'),
            'candidate_ratings_column_name': feature_conf.get('candidate_ratings_column'),
            'history_date_column_name': feature_conf.get('history_date_column'),
            'candidate_date_column_name': feature_conf.get('candidate_date_column'),
            'cut_off_time': cut_off_time,
            'history_length': history_length,
            'num_rerank': num_rerank,
            'token_per_item': token_per_item
        }

        # 分别创建训练和验证数据集
        train_dataset = DatasetMusicLonger(
            ratings_file=get_format_csv(data_dir, train_data_path),
            is_train=True,
            **dataset_params
        )
        eval_dataset = DatasetMusicLonger(
            ratings_file=get_format_csv(data_dir, valid_data_path),
            is_train=False,
            **dataset_params
        )
    elif dataset == 'Game':
        token_per_item = 1 if feature_conf.get("fuse_ia", True) else 2
        cut_off_time = feature_conf.get("cut_off_time", None)
        if cut_off_time is None:
            raise ValueError("cut_off_time in config should be set as date for AG.")
        train_dataset = DatasetAG(
            ratings_file=get_format_csv(data_dir, train_data_path),
            ignore_last_n=0,
            chronological=chronological,
            is_ads=feature_conf.get('time_desc', True),
            sep=';',
            rank=rank,
            world_size=world_size,
            history_items_key=feature_conf.get("history_items_key"),
            candidate_items_key=feature_conf.get("candidate_items_key"),
            history_feature_columns=feature_conf.get('history_item_feature_columns'),
            candidate_feature_columns=feature_conf.get('candidate_item_feature_columns'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            history_ratings_column_name=feature_conf.get('history_ratings_column'),
            candidate_ratings_column_name=feature_conf.get('candidate_ratings_column'),
            history_timestamps_column_name=feature_conf.get('history_timestamps_column'),
            candidate_timestamps_column_name=feature_conf.get('candidate_timestamps_column'),
            history_date_column_name=feature_conf.get('history_date_column'),
            candidate_date_column_name=feature_conf.get('candidate_date_column'),
            fill_dates=feature_conf.get('fill_dates', False),
            is_train=True,
            cut_off_time=cut_off_time,
            history_length=history_length,
            num_rerank=num_rerank,
            token_per_item=token_per_item,
            use_jagged_data=False,
            use_dynamic_padding=use_dynamic_padding,
            padding_side_origin=padding_side_origin,
            padding_side=padding_side
        )
        eval_dataset = DatasetAG(
            ratings_file=get_format_csv(data_dir, valid_data_path),
            ignore_last_n=0,
            chronological=chronological,
            is_ads=feature_conf.get('time_desc', True),
            sep=';',
            rank=rank,
            world_size=world_size,
            history_items_key=feature_conf.get("history_items_key"),
            candidate_items_key=feature_conf.get("candidate_items_key"),
            history_feature_columns=feature_conf.get('history_item_feature_columns'),
            candidate_feature_columns=feature_conf.get('candidate_item_feature_columns'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            history_ratings_column_name=feature_conf.get('history_ratings_column'),
            candidate_ratings_column_name=feature_conf.get('candidate_ratings_column'),
            history_timestamps_column_name=feature_conf.get('history_timestamps_column'),
            candidate_timestamps_column_name=feature_conf.get('candidate_timestamps_column'),
            history_date_column_name=feature_conf.get('history_date_column'),
            candidate_date_column_name=feature_conf.get('candidate_date_column'),
            fill_dates=feature_conf.get('fill_dates', False),
            is_train=False,
            cut_off_time=cut_off_time,
            history_length=history_length,
            num_rerank=num_rerank,
            token_per_item=token_per_item,
            use_dynamic_padding=use_dynamic_padding,
            padding_side_origin=padding_side_origin,
            padding_side=padding_side
        )
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    all_item_ids = [x + 1 for x in range(Const.EXPECTED_NUM_UNIQUE_ITEMS)]
    max_item_id = Const.EXPECTED_NUM_UNIQUE_ITEMS

    return RecoDataset(
        max_sequence_length=max_sequence_length,
        num_unique_items=Const.EXPECTED_NUM_UNIQUE_ITEMS,
        max_item_id=max_item_id,
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        use_dynamic_padding=use_dynamic_padding,
    )
