import ast
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple
import random
import pandas as pd
import numpy as np
import torch
import traceback
import json
from collections import OrderedDict
from torch.utils.data import IterableDataset


class MultiCSVIterator:
    """用于读取csv文件的迭代器。"""

    def __init__(self, file_paths: List[str], sep: str, rank: int, world_size: int, column_names, inner_delim,
                 itemid_column_name='item_id', file_format='csv', split_chunks_by_worker_id=True, chunksize=10 ** 6,
                 is_train=None) -> None:
        """
        初始化MultiCSVIterator。

        :param file_paths: csv文件路径列表。
        :param sep: csv文件的分隔符。
        :param rank: 用于分布式训练的rank，0表示主进程。
        :param world_size: 分布式并行总进程数。
        :param column_names: csv文件的列名。
        :param inner_delim: 内部分隔符，用于处理列中的多个值。
        :param itemid_column_name: 表示物品ID的列名，默认为'item_id'。
        """

        self.file_paths = file_paths
        self.sep = sep
        self.rank = rank
        self.world_size = world_size

        self.current_file_index = 0
        self.chunk_iterator = iter([])
        self.chunk_data = None
        self.current_data_index = 0
        self.total_read_count = 0

        self.column_names = column_names
        self.inner_delim = inner_delim

        self.itemid_column_name = itemid_column_name
        self.worker_info = torch.utils.data.get_worker_info()
        self.file_format = file_format
        self.split_chunks_by_worker_id = split_chunks_by_worker_id
        self.increment = 1 if self.split_chunks_by_worker_id else self.world_size
        self.chunk_size = chunksize
        self.is_train = is_train

    def __iter__(self):
        return self

    def read_next_file(self):
        if self.current_file_index >= len(self.file_paths):
            logging.info('worker_id %s finished data iteration ', self.worker_info)
            raise StopIteration

        if self.file_format == 'orc':
            self.chunk_iterator = pd.read_orc(self.file_paths[self.current_file_index])
            self.chunk_iterator = self.chunk_iterator.sample(frac=1).reset_index(drop=True)
        elif self.file_format == 'csv':
            if self.split_chunks_by_worker_id:
                self.chunk_iterator = pd.read_csv(self.file_paths[self.current_file_index], sep=self.sep)
            else:
                self.chunk_iterator = pd.read_csv(self.file_paths[self.current_file_index], sep=self.sep,
                                                  chunksize=self.chunk_size)
        self.current_file_index += 1

    def get_next_chunk(self):
        self.read_next_file()
        self.total_read_count = len(self.chunk_iterator) // self.world_size
        read_start_idx = self.total_read_count * self.rank
        self.chunk_data = self.chunk_iterator[read_start_idx: read_start_idx + self.total_read_count]
        del self.chunk_iterator
        self.current_data_index = 0

    def __next__(self):
        if self.chunk_data is None:
            self.get_next_chunk()

        if self.current_data_index < self.total_read_count:
            data = self.chunk_data.iloc[self.current_data_index]
            self.current_data_index += self.increment
            return data
        else:
            self.chunk_data = None
            return self.__next__()


class DatasetMusicV2LONGER(IterableDataset):

    def __init__(
            self,
            dataset_config_dir,
            user_feat_config: dict,
            item_feat_config: dict,
            seq_feat_config: dict,
            dataset_train_dir: str = None,
            dataset_valid_dir: str = None,
            rank: int = 0,
            world_size: int = 1,
            is_train: bool = True,
            file_format='orc',
            sep: str = ',',
            exclude_features=None,
            num_rerank=256,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        """
        初始化DatasetV8。

        :param user_feat_config: 用户特征配置。
        :param item_feat_config: 商品特征配置。
        :param seq_feat_config: 序列特征配置。
        :param dataset_train_dir: 训练文件路径。
        :param dataset_valid_dir: 测试文件路径。
        :param chronological: 时序特征是否按正序排列。
        :param rank: 用于分布式训练的rank。
        :param world_size: 用于分布式训练的世界大小。
        :param is_train: 是否为训练数据。
        :parma file_format: 文件的格式
        """

        self.rank = rank
        self.world_size = world_size
        self.sep = sep
        self.is_train = is_train
        self.exclude_features = exclude_features

        if is_train and dataset_train_dir is None:
            raise ValueError('dataset_train_dir should be configured when it is training')
        if not is_train and dataset_valid_dir is None:
            raise ValueError('dataset_valid_dir should be configured when it is not training')

        if is_train:
            train_files = os.listdir(dataset_train_dir)
            random.shuffle(train_files)
            self.files = [dataset_train_dir + '/' + f for f in train_files if f.startswith("part")]
        else:
            self.num_rerank = num_rerank
            valid_files = os.listdir(dataset_valid_dir)
            random.shuffle(valid_files)
            self.files = [dataset_valid_dir + '/' + f for f in valid_files if
                          (f.startswith("part") or f.startswith("output"))]

        self.multi_csv_iterator = None
        self.file_format = file_format

        self.seq_feat_config = seq_feat_config
        self.item_feat_config = item_feat_config
        self.user_feat_config = user_feat_config

    def init_multi_csv_iterator(self) -> None:
        if not self.multi_csv_iterator:
            worker_info = torch.utils.data.get_worker_info()
            self.multi_csv_iterator = MultiCSVIterator(self.files, self.sep, self.rank, self.world_size,
                                                       None, self.sep,
                                                       itemid_column_name=None, file_format=self.file_format,
                                                       split_chunks_by_worker_id=True,
                                                       is_train=self.is_train)

    def __iter__(self):
        # 当前df的数据索引
        it = map(self.load_item, self.multi_csv_iterator)
        return it

    def pad_or_truncate_item_feature(self, feat_value: torch.tensor, max_length: int, dtype):
        if dtype == "int" or dtype == "con":
            num_valid_test_items = feat_value.size(-1)
        else:
            num_valid_test_items = feat_value.size(0)
        if num_valid_test_items < max_length:
            if dtype == "int" or dtype == "con":
                # pad (1, N) to (1, max_length), 在后面补0
                feat_value = torch.nn.functional.pad(feat_value, (0, max_length - num_valid_test_items))
            else:
                # pad (N, M) to (max_length, M), 在后面补0
                feat_value = torch.nn.functional.pad(feat_value, (0, 0, 0, max_length - num_valid_test_items))
        else:
            if dtype == "int" or dtype == "con":
                # pad (1, N) to (1, max_length), 在后面补0
                feat_value = feat_value[:, :max_length]
            else:
                feat_value = feat_value[:max_length, :]
        return feat_value, num_valid_test_items

    def load_item(self, data) -> Dict[str, torch.Tensor]:
        """
        加载单个数据项。

        :param data: 单个数据项。
        :return: 处理后的数据字典
        """
        ret = {}
        for key, value in data.items():
            if key in self.exclude_features:
                continue
            if key == "user_id":
                # user_id 不使用原始的字符串，填0
                value = torch.tensor(abs(hash(str(value))) % 1e+9, dtype=torch.int64)
            else:
                # 处理用户、商品、序列特征
                if self.is_train:
                    # 训练集的数据处理逻辑
                    # 对于其他值，都是int或者float列表
                    value = value.tolist()
                    if len(value) > 1:
                        # 多值特征,没有浮点数
                        value = torch.tensor(value, dtype=torch.int64)
                    else:
                        # 单值特征，可能浮点数或者整数

                        if isinstance(value[0], int):
                            if key in self.item_feat_config:
                                value = torch.tensor(value, dtype=torch.int64)
                            else:
                                value = torch.tensor(value[0], dtype=torch.int64)
                        elif isinstance(value[0], float):
                            if key in self.item_feat_config:
                                value = torch.tensor(value, dtype=torch.float32)
                            else:
                                value = torch.tensor(value[0], dtype=torch.float32)
                        else:
                            logging.error("please check the data type value of %s, which is %s", key, value)
                else:

                    # 测试集的数据处理逻辑
                    if key in self.user_feat_config:
                        # 如果是用户特征
                        if self.user_feat_config[key]["dtype"] == "int":
                            # 单值特征
                            value = torch.tensor(value, dtype=torch.int64)
                        elif self.user_feat_config[key]["dtype"] == "pref" or self.user_feat_config[key][
                            "dtype"] == "multi":
                            # 偏好特征
                            value = torch.tensor([int(x) for x in value.split(",")], dtype=torch.int64)
                    elif key in self.item_feat_config:
                        if self.item_feat_config[key]["dtype"] == "int":
                            # 单值
                            value = torch.tensor(value, dtype=torch.int64).reshape(1, -1)
                        elif self.item_feat_config[key]["dtype"] == "con":
                            # 连续
                            value = torch.tensor(value, dtype=torch.float32).reshape(1, -1)
                        elif self.item_feat_config[key]["dtype"] == "multi":
                            value = torch.tensor([int(x) for x in value], dtype=torch.int64).reshape(1, -1)
                        value, _ = self.pad_or_truncate_item_feature(value, self.num_rerank,
                                                                     self.item_feat_config[key]["dtype"])

                    else:
                        if "seq" in key:
                            # 序列特征
                            value = torch.tensor([int(x) for x in value.split(",")], dtype=torch.int64)
                        elif key == "label":
                            value = torch.tensor(value, dtype=torch.float32).reshape(1, -1)
                            value, valid_length = self.pad_or_truncate_item_feature(value, self.num_rerank, "con")
                        else:
                            continue
            ret[key] = value
        if self.is_train:
            ret["valid_items"] = torch.tensor(1, dtype=torch.int64)
        else:
            ret["valid_items"] = torch.tensor(valid_length, dtype=torch.int64)
        return ret
