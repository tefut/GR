import os
import ast
from itertools import compress
import traceback

import torch
import logging
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset
from typing import Dict, List, Optional, Tuple, Union, Any

from modeling.generic.utils.constants import DEFAULT_DATA_LEN


class MultiFileIterator:
    """An iterator used to read orc file formats."""

    def __init__(
            self,
            file_paths: List[str],
            sep: str,
            rank: int,
            world_size: int,
            column_names: List[str],
            inner_delim: str,
            itemid_column_name: str = "song_id",
            file_format: str = "orc",
            split_chunks_by_worker_id: bool = False,
            chunksize: int = 10 ** 5
    ) -> None:
        """
        MultiFileIterator

        :param file_paths: List of file paths.
        :param sep: CSV delimiter (only used for CSV files).
        :param rank: Distributed training rank, 0 indicates main process.
        :param world_size: Number of distributed parallel processes.
        :param column_names: Names of the columns to load.
        :param inner_delim: Delimiter for list columns (only used for list columns stored as string).
        :param itemid_column_name: Column name of item ID.
        :param file_format: File format of data files (currently supported: csv, parquet, orc).
        :param split_chunks_by_worker_id: If chunks are split by worker ID (global).
        :param chunksize: The number of rows read per chunk.
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

    def __iter__(self):
        return self

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

    def read_next_file(self):
        if self.current_file_index >= len(self.file_paths):
            logging.info('worker_id %s finished data iteration', self.worker_info.id)
            raise StopIteration

        if self.file_format == 'orc':
            df = pd.read_orc(self.file_paths[self.current_file_index])
            self.chunk_iterator = iter([df])
        elif self.file_format == 'csv':
            df = pd.read_csv(self.file_paths[self.current_file_index])
            self.chunk_iterator = iter([df])
        elif self.file_format == 'parquet':
            df = pd.read_parquet(self.file_paths[self.current_file_index])
            self.chunk_iterator = iter([df])
        else:
            raise NotImplementedError("Unsupported file format %s", self.file_format)

        self.current_file_index += 1

    def get_next_chunk(self) -> None:
        if self.split_chunks_by_worker_id:
            self.read_next_file()
            self.total_read_count = len(self.chunk_iterator) // self.world_size
            read_start_idx = self.total_read_count * self.rank
            self.chunk_data = self.chunk_iterator[read_start_idx: read_start_idx + self.total_read_count]
            del self.chunk_iterator
            self.current_data_index = 0
            return None
        else:
            try:
                self.chunk_data = self.chunk_iterator.__next__()
                self.total_read_count = len(self.chunk_data) - len(self.chunk_data) % self.world_size
                self.current_data_index = self.rank
                return None
            except StopIteration:
                # if get next chunk fails, read next file
                self.read_next_file()
                return self.__next__()


class DatasetMusicLonger(IterableDataset):
    """用于处理按时间逆序排列的数据集。"""

    def __init__(
            self,
            ratings_file: str,
            ignore_last_n: int = 0,
            shift_id_by: int = 0,
            chronological: bool = True,
            sample_ratio: float = 1.0,
            is_ads: bool = False,
            sep: str = ',',
            rank: int = 0,
            world_size: int = 1,
            history_feature_columns=None,
            candidate_feature_columns=None,
            user_feature_columns=None,
            candidate_items_key='song_id',
            candidate_ratings_column_name='label',
            history_date_column_name='play_song_datediff_seq',
            history_hour_column_name='play_song_hour_seq',
            candidate_date_column_name='oper_date',
            inner_delim: str = ',',
            file_format: str = 'orc',
            is_train: bool = True,
            cut_off_time: int = None,
            history_length=200,
            num_rerank=1,
            use_repadding: bool = False,
            token_per_item: int = 2
    ) -> None:
        super().__init__()
        """
        初始化DatasetMusicLonger。
        :param ratings_file: 文件路径。
        :param ignore_last_n: 忽略最后n个数据项。
        :param shift_id_by: ID偏移量。
        :param chronological: 默认为True, 处理后按时间正序排列。
        :param sample_ratio: 采样比例。
        :param is_ads: 是否为广告数据。
        :param sep: 分隔符。
        :param rank: 用于分布式训练的rank。
        :param world_size: 用于分布式训练的世界大小。
        :param history_feature_columns:
        :param candidate_feature_columns: 注意candidate使用的feature肯能和history不一样 
        :param user_feature_columns: 用户特征。
        :param itemid_column_name: 物品ID列名。
        :param ratings_column_name: 评分列名。
        :param timestamps_column_name: 时间戳列名。
        :param inner_delim: 内部分隔符。
        :param is_train: 是否为训练数据。
        :param cut_off_time: 训练/验证切分时间戳, 默认为093000对应的时间戳。
        """
        if history_feature_columns is None:
            raise ValueError('history_feature_columns should not be None')
        if candidate_feature_columns is None:
            raise ValueError('candidate_feature_columns should not be None')
        if user_feature_columns is None:
            raise ValueError('nonseq_columns should not be None')

        self.rank = rank
        self.world_size = world_size

        # 获取数据文件
        files = self.get_data_files(ratings_file)
        self.files = [ratings_file + '/' + f for f in files]

        self.file_format = file_format
        self.current_ratings_frame_len = 0
        self.current_ratings_frame_idx = 0
        self.data_idx = 0
        self.ratings_frame = None

        self._max_history_length = history_length
        self._ignore_last_n: int = ignore_last_n
        self._cache = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self._sample_ratio: float = sample_ratio

        self.history_feature_columns = history_feature_columns
        self.candidate_feature_columns = candidate_feature_columns
        self.user_feature_columns = user_feature_columns
        self.history_feature_column_names = [k for k, _ in history_feature_columns.items()]
        self.candidate_feature_column_names = [k for k, _ in candidate_feature_columns.items()]
        self.user_feature_column_names = [k for k, _ in user_feature_columns.items()]
        self.candidate_items_key = candidate_items_key
        self.candidate_ratings_column_name = candidate_ratings_column_name
        self.history_date_column_name = history_date_column_name
        self.candidate_date_column_name = candidate_date_column_name
        self.history_hour_column_name = history_hour_column_name

        extra_column_names = [
            self.candidate_ratings_column_name,
            self.history_date_column_name,
            self.candidate_date_column_name,
            self.history_hour_column_name
        ]
        self.column_names = (
                self.history_feature_column_names +
                self.candidate_feature_column_names +
                self.user_feature_column_names +
                extra_column_names
        )

        self.inner_delim = inner_delim
        self.cut_off_time = cut_off_time
        self.is_train = is_train

        self.is_ads = is_ads
        self.sep = sep
        self.multi_csv_iterator = None
        self._max_candidate_num: int = num_rerank

        self.use_repadding = use_repadding
        if self.use_repadding and rank == 0:
            logging.info('Using sequence expand by repadding')

        self.token_per_item = token_per_item

    def __len__(self):
        return DEFAULT_DATA_LEN

    def __iter__(self):
        self.init_multi_csv_iterator()
        # 当前df的数据索引
        it = map(self.load_item, self.multi_csv_iterator)
        return it

    @staticmethod
    def get_data_files(file_path: str) -> List[str]:
        try:
            files = os.listdir(file_path)
        except FileNotFoundError as fnf_error:
            logging.error('No such file or directory: %s', file_path)
            raise fnf_error
        except NotADirectoryError as na_dir_error:
            logging.error('Not a directory: %s', file_path)
            raise na_dir_error
        except OSError as os_error:
            logging.error('OS error: %s', file_path)
            raise os_error
        return files

    @staticmethod
    def _truncate_or_pad_seq(
            y: List[Union[int, List[int], float]],
            target_len: int,
            chronological: bool,
            max_len_per_item: int = 1
    ) -> List[Union[int, List[int], float]]:
        """
        将序列 y 调整到长度为 target_len。如果 len_per_item > 0，则假定 y 中的每个元素本身是长度可变的列表，
        先对齐（truncate 或 pad）每个内层列表到长度 len_per_item，再对齐外层列表到长度 target_len；否则仅对齐外层列表。

        参数:
        - y: 原始序列，要么是 List[int]（len_per_item=0），要么是 List[List[int]]（len_per_item>0）。
        - target_len: 目标序列长度。外层长度会被截断或补齐到此值。
        - chronological: 当外层长度大于 target_len 时，决定保留头部还是尾部。
                        True 表示保留最靠近"当前"的末尾 target_len 个元素，False 表示保留最前面的 target_len 个元素。
        - max_len_per_item: 如果 > 0，表示要先把 y 中每个元素（当作列表）对齐到长度 len_per_item，再再做外层对齐。
                        如果 =0，则认为 y 中元素是单个整数，直接对齐外层长度。

        返回值:
        - List[Union[int, List[int]]]: 长度恰为 target_len 的序列；当 len_per_item>0 时，内层列表也固定为长度 len_per_item，
        padding 用 0 填充，截断时取头部（默认）。
        """
        # —— 1. 如果需要对齐内层列表，先对齐每个内层
        if max_len_per_item > 1:
            aligned_inner = []
            for elem in y:
                # 如果原始 elem 不是 list，也把它当作长度 1 的列表处理
                if not isinstance(elem, list):
                    current = [elem]
                else:
                    current = elem.copy()

                # 对齐到 len_per_item: 先截断或填 0
                if len(current) < max_len_per_item:
                    # padding
                    current = current + [0] * (max_len_per_item - len(current))
                elif len(current) > max_len_per_item:
                    # 截断：只保留最前面的 len_per_item 个元素
                    current = current[:max_len_per_item]

                aligned_inner.append(current)
            y = aligned_inner

        # —— 2. 外层长度对齐（truncate or pad）
        y_len = len(y)
        if target_len == 0:
            y = []
        elif y_len < target_len:
            if max_len_per_item > 1:
                pad_unit = [0] * max_len_per_item
                pads = [pad_unit for _ in range(target_len - y_len)]
            else:
                pads = [0] * (target_len - y_len)
            y = y + pads

        elif y_len > target_len:
            if chronological:
                # 保留末尾 target_len 个元素
                y = y[-target_len:]
            else:
                # 保留最前面 target_len 个元素
                y = y[:target_len]

        if len(y) != target_len:
            raise ValueError(f"对齐后序列长度不等于 target_len（{target_len}），got {len(y)}")

        return y

    @staticmethod
    def _process_user_id(user_id_raw: Union[str, bytes, np.ndarray, int, float]) -> int:
        """
        处理用户ID，转换为64位整数

        :param user_id_raw: 原始用户ID数据
        :return: 处理后的用户ID整数
        """
        try:
            # 处理numpy数组
            if isinstance(user_id_raw, np.ndarray):
                # 安全地处理numpy数组
                if user_id_raw.size == 1:
                    user_id_raw = user_id_raw.item()
                elif user_id_raw.size > 1:
                    # 如果有多个元素，取第一个
                    user_id_raw = user_id_raw.flat[0]
                else:
                    # 如果数组为空，使用默认值
                    user_id_raw = ""

            # 如果是字节类型，先解码为字符串
            if isinstance(user_id_raw, bytes):
                user_id = user_id_raw.decode('utf-8')  # 将bytes转换为字符串
            else:
                user_id = str(user_id_raw)

            # 移除可能的前缀（如 b' 前缀说明符）
            if user_id.startswith("b'") and user_id.endswith("'"):
                user_id = user_id[2:-1]

            # 确保是有效的16进制字符串
            # 移除可能的 '0x' 前缀
            if user_id.startswith('0x'):
                user_id = user_id[2:]

            # 清理可能的转义字符
            user_id = user_id.replace('\\', '')

            # 只保留有效的16进制字符
            import re
            user_id = re.sub(r'[^0-9a-fA-F]', '', user_id)

            # 只保留前15个字符以避免torch.int64溢出
            user_id_truncated = user_id[0:min(15, len(user_id))]

            # 转换为整数（16进制）
            if user_id_truncated:  # 确保不为空
                uid = int(user_id_truncated, 16)
            else:
                uid = 0

        except Exception as e:
            # 如果发生错误，打印调试信息并设置默认值
            logging.info("Error processing user_id - raw data: %s, type: %s, error: %s", user_id_raw,
                         type(user_id_raw), e)
            uid = 0  # 设置默认值

        return uid

    def init_multi_csv_iterator(self) -> None:
        if not self.multi_csv_iterator:
            self.multi_csv_iterator = MultiFileIterator(self.files, self.sep, self.rank, self.world_size,
                                                        self.column_names,
                                                        self.inner_delim,
                                                        file_format=self.file_format,
                                                        itemid_column_name=self.candidate_items_key,
                                                        split_chunks_by_worker_id=False)

    def _process_feature(self, x: Union[np.ndarray, str, int, float, list],
                         ignore_last_n: int, shift_id_by: int = 0,
                         sampling_kept_mask: Optional[List[bool]] = None,
                         feature_dtype: Optional[str] = None) -> Tuple[List[Union[int, float]], int]:
        """
        处理单个特征数据

        :param x: 原始特征数据
        :param ignore_last_n: 忽略最后n个元素
        :param shift_id_by: ID偏移量
        :param feature_dtype: 特征数据类型
        :return: 处理后的特征列表
        """
        try:
            if feature_dtype == "multi":
                if hasattr(x, "ndim") and x.ndim == 1:
                    x = np.array([x])
            # 处理numpy数组
            if isinstance(x, np.ndarray):
                if x.ndim == 0:
                    y_list = [x.item()]
                else:
                    y_list = x.tolist()
            # 处理字符串数据
            else:
                if isinstance(x, str):
                    x = x.split(',')
                    if '.' in x[0]:
                        y_list = list(map(float, x))
                    else:
                        y_list = list(map(int, x))
                else:
                    y_list = [x]

            # 时间顺序处理
            if not self._chronological:  # 如果需要逆序
                y_list.reverse()

            # 忽略最近的n个数据
            if ignore_last_n > 0:
                y_list = y_list[:-ignore_last_n] if len(y_list) > ignore_last_n else []
                if not y_list:
                    y_list = [0]
            if sampling_kept_mask is not None:
                y_list = [x for x, kept in zip(y_list, sampling_kept_mask) if kept]

            y_len = len(y_list)

            if shift_id_by > 0:
                y_list = [x + shift_id_by for x in y_list]

            return y_list, y_len
        except AttributeError as e:
            logging.error("AttributeError: %s", str(e))
            raise e
        except ValueError as e:
            logging.error("ValueError: %s", str(e))
            raise e
        except Exception as e:
            logging.error("Other Exception: %s", str(e))
            raise e

    def _process_features_common(
            self,
            data: Dict[str, Any],
            column_names: List[str],
            feature_columns: Dict[str, Dict],
            max_len_per_multival: Dict[str, int],
            ignore_last_n: int,
            shift_id_by: int
    ) -> Tuple[Dict[str, List[Union[int, float]]], Dict[str, int]]:
        """
        处理特征的通用函数
        :param data: 原始数据字典
        :param column_names: 需要处理的列名列表
        :param feature_columns: 特征列配置字典
        :param max_len_per_multival: 多值特征最大长度配置
        :param ignore_last_n: 忽略最后n个元素
        :param shift_id_by: ID偏移量
        :return: 处理后的特征列表和长度字典
        """
        feature_list = {}
        feature_len = {}
        for column_name in column_names:
            feature_dtype = feature_columns.get(column_name).get("dtype")
            seq_feature, seq_len = self._process_feature(
                data[column_name],
                ignore_last_n=ignore_last_n,
                shift_id_by=shift_id_by,
                feature_dtype=feature_dtype
            )
            feature_list[column_name] = seq_feature
            feature_len[column_name] = seq_len
            feat_max_len = feature_columns.get(column_name).get("max_len", 1)
            max_len_per_multival[column_name] = feat_max_len
        return feature_list, feature_len

    def _get_hist_features(self, data: Dict[str, Any], max_len_per_multival: Dict[str, int]) -> Dict[str, Any]:
        """
        处理历史序列特征
        :param data: 原始数据字典
        :param max_len_per_multival: 多值特征的最大长度配置
        :return: 历史序列信息字典
        """
        # 处理历史特征列
        hist_feature_list, hist_feature_len = self._process_features_common(
            data=data,
            column_names=self.history_feature_column_names,
            feature_columns=self.history_feature_columns,
            max_len_per_multival=max_len_per_multival,
            ignore_last_n=self._ignore_last_n,
            shift_id_by=self._shift_id_by
        )

        # 处理历史时间特征
        hist_dates, _ = self._process_feature(
            data[self.history_date_column_name],
            ignore_last_n=self._ignore_last_n,
            shift_id_by=0
        )
        # 序列长度截断和padding处理
        hist_length = min(len(hist_dates), self._max_history_length)
        # 构造截断参数
        hist_trunc_params = {
            'target_len': self._max_history_length,
            'chronological': True
        }
        # 对历史序列进行截断或padding
        hist_dates = self._truncate_or_pad_seq(hist_dates, **hist_trunc_params)

        # 处理历史特征序列
        for k, v in hist_feature_list.items():
            hist_trunc_params['max_len_per_item'] = max_len_per_multival.get(k, 1)
            v = self._truncate_or_pad_seq(v, **hist_trunc_params)
            hist_feature_list[k] = v
        return {
            'hist_length': hist_length,
            'hist_feature_list': hist_feature_list,
            'hist_feature_len': hist_feature_len,
            'hist_dates': hist_dates,
            'hist_ratings': [0] * len(hist_dates)
        }

    def _get_cand_features(self, data: Dict[str, Any], max_len_per_multival: Dict[str, int]) -> Dict[str, Any]:
        """
        处理候选序列特征
        :param data: 原始数据字典
        :param max_len_per_multival: 多值特征的最大长度配置
        :return: 候选序列信息字典
        """
        # 处理候选特征列
        candidate_feature_list, candidate_feature_len = self._process_features_common(
            data=data,
            column_names=self.candidate_feature_column_names,
            feature_columns=self.candidate_feature_columns,
            max_len_per_multival=max_len_per_multival,
            ignore_last_n=self._ignore_last_n,
            shift_id_by=self._shift_id_by
        )

        # 处理候选评分特征
        candidate_ratings, candidate_ratings_len = self._process_feature(
            data[self.candidate_ratings_column_name],
            ignore_last_n=self._ignore_last_n,
            shift_id_by=0
        )

        if candidate_ratings and isinstance(candidate_ratings[0], str):
            if '.' in candidate_ratings[0]:
                candidate_ratings = [float(x) for x in candidate_ratings]
        candidate_ratings = list(map(int, candidate_ratings))

        # 处理候选特征的交错展开
        def candidate_feature_interleave(candidate_feature: list, n: int = 2) -> list:
            length = len(candidate_feature)
            return [candidate_feature[i // n] for i in range(length * n)]

        # 候选特征序列处理
        max_candidate_num = self._max_candidate_num
        candidate_trunc_params = {
            'target_len': max_candidate_num,
            'chronological': True
        }
        # 对候选序列进行截断或padding
        candidate_ratings = self._truncate_or_pad_seq(candidate_ratings, **candidate_trunc_params)
        candidate_ratings = candidate_feature_interleave(candidate_ratings, self.token_per_item)
        # 处理候选特征序列
        for k, v in candidate_feature_list.items():
            candidate_trunc_params['max_len_per_item'] = max_len_per_multival.get(k, 1)
            v = self._truncate_or_pad_seq(v, **candidate_trunc_params)
            v = candidate_feature_interleave(v, self.token_per_item)
            candidate_feature_list[k] = v
        # 处理训练数据标签
        labels = candidate_ratings  # candidate_action_type本身只有0曝光1下载，直接可以用做ground_truth

        # 处理模型权重 - 注意长度应该是交错展开后的长度
        loss_weights = [1] * (max_candidate_num * self.token_per_item)

        # 间隔处loss weights为0，但是rating需要在embedding内
        if self.use_repadding:  # 间隔处是补零位
            candidate_ratings = [0 if x == -1 else x for x in candidate_ratings]
        return {
            'candidate_feature_list': candidate_feature_list,
            'candidate_feature_len': candidate_feature_len,
            'candidate_ratings': candidate_ratings,
            'labels': labels,
            'loss_weights': loss_weights
        }

    def _get_user_features(self, data: Dict[str, Any], max_len_per_multival: Dict[str, int]) -> Dict[str, Any]:
        """
        处理用户非序列特征
        :param data: 原始数据字典
        :param max_len_per_multival: 多值特征的最大长度配置
        :return: 用户特征信息字典
        """
        # 处理用户特征列
        non_seq_feature, non_seq_feature_len = self._process_features_common(
            data=data,
            column_names=self.user_feature_column_names,
            feature_columns=self.user_feature_columns,
            max_len_per_multival=max_len_per_multival,
            ignore_last_n=self._ignore_last_n,
            shift_id_by=self._shift_id_by
        )

        # 对用户特征进行截断或padding处理（用户特征通常需要padding到固定长度1）
        for k, v in non_seq_feature.items():
            v = self._truncate_or_pad_seq(
                v,
                target_len=1,
                chronological=True,
                max_len_per_item=max_len_per_multival.get(k, 1)
            )
            non_seq_feature[k] = v

        return {
            'non_seq_feature': non_seq_feature,
            'non_seq_feature_len': non_seq_feature_len
        }

    def load_item(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        加载单个数据项。

        :param data: 单个数据项字典
        :return: 处理后的数据项张量字典
        """
        # 读取特征
        max_len_per_multival: Dict[str, int] = {}  # 每个特征的最大长度，通过config读取，单值特征为1，多值特征为n
        # 读取用户特征
        user_features = self._get_user_features(data, max_len_per_multival)
        # 读取历史部分
        hist_features = self._get_hist_features(data, max_len_per_multival)
        # 读取候选部分
        cand_features = self._get_cand_features(data, max_len_per_multival)
        # 处理用户ID
        uid = self._process_user_id(data['user_id'])

        # 构建返回结果
        ret: Dict[str, torch.Tensor] = {
            "uid": torch.tensor(uid, dtype=torch.int64),
            "candidate_action_type": torch.tensor(cand_features['candidate_ratings'], dtype=torch.int64),
            "loss_weights": torch.tensor(cand_features['loss_weights'], dtype=torch.float32),
            "label": torch.tensor(cand_features['labels'], dtype=torch.int64),
            self.history_date_column_name: torch.tensor(hist_features['hist_dates'], dtype=torch.int64),
        }

        for k, v in user_features['non_seq_feature'].items():
            try:
                ret[k] = torch.tensor(v, dtype=torch.int64)
            except Exception as e:
                logging.info("Error processing non_seq_feature: %s, value: %s", k, v)
        for k, v in hist_features['hist_feature_list'].items():
            try:
                ret[k] = torch.tensor(v, dtype=torch.int64)
            except Exception as e:
                logging.info("Error processing history_feature: %s, value: %s", k, v)
        for k, v in cand_features['candidate_feature_list'].items():
            if isinstance(v[0], (np.float32, np.float64, float)):
                try:
                    ret[k] = torch.tensor(v, dtype=torch.float32)
                except Exception as e:
                    logging.info("Error processing candidate_feature: %s, value: %s", k, v)
            else:
                try:
                    ret[k] = torch.tensor(v, dtype=torch.int64)
                except Exception as e:
                    logging.info("Error processing candidate_feature: %s, value: %s", k, v)

        return ret


class DatasetAG(IterableDataset):
    """用于处理按时间逆序排列的数据集。"""

    def __init__(
            self,
            ratings_file: str,
            ignore_last_n: int,
            shift_id_by: int = 0,
            chronological: bool = True,
            sample_ratio: float = 1.0,
            is_ads: bool = False,
            sep: str = ',',
            rank: int = 0,
            world_size: int = 1,
            history_items_key=None,
            candidate_items_key=None,
            history_feature_columns=None,
            candidate_feature_columns=None,
            nonseq_columns=None,
            itemid_column_name='candidate_app_id',
            history_ratings_column_name='history_action_type',
            candidate_ratings_column_name='candidate_action_type',
            history_timestamps_column_name="history_timestamps",
            candidate_timestamps_column_name='candidate_timestamps',
            history_date_column_name='history_date',
            candidate_date_column_name='candidate_date',
            fill_dates=True,
            inner_delim: str = ' ',
            is_train: bool = True,
            cut_off_time: int = None,
            history_length=400,
            num_rerank=400,
            use_repadding: bool = False,
            token_per_item: int = 2,
            use_jagged_data: bool = False,
            use_dynamic_padding: bool = False,
            padding_side_origin: str = "left",
            padding_side: str = "right"
    ) -> None:
        super().__init__()
        """
        初始化DatasetAG。

        :param ratings_file: 文件路径。
        :param ignore_last_n: 忽略最后n个数据项。
        :param shift_id_by: ID偏移量。
        :param chronological: 默认为True, 处理后按时间正序排列。
        :param sample_ratio: 采样比例。
        :param is_ads: 是否为广告数据。
        :param sep: 分隔符。
        :param rank: 用于分布式训练的rank。
        :param world_size: 用于分布式训练的世界大小。
        :param history_feature_columns:
        :param candidate_feature_columns: 注意candidate使用的feature肯能和history不一样 
        :param nonseq_columns: 非序列列名，这里指人特征名。
        :param itemid_column_name: 物品ID列名。
        :param ratings_column_name: 评分列名。
        :param timestamps_column_name: 时间戳列名。
        :param inner_delim: 内部分隔符。
        :param is_train: 是否为训练数据。
        :param cut_off_time: 训练/验证切分时间戳, 默认为093000对应的时间戳。
        """

        if history_feature_columns is None:
            raise ValueError('history_feature_columns should not be None')
        if candidate_feature_columns is None:
            raise ValueError('candidate_feature_columns should not be None')
        if nonseq_columns is None:
            raise ValueError('nonseq_columns should not be None')

        self.rank = rank
        self.world_size = world_size

        try:
            files = os.listdir(ratings_file)
        except FileNotFoundError as fnf_error:
            logging.error('No such file or directory: %s', ratings_file)
            raise fnf_error
        except NotADirectoryError as na_dir_error:
            logging.error('Not a directory: %s', ratings_file)
            raise na_dir_error
        except OSError as os_error:
            logging.error('OS error: %s', ratings_file)
            raise os_error

        self.files = [ratings_file + '/' + f for f in files
                      if f.endswith('.csv') or f.endswith('.parquet')]

        self.current_ratings_frame_len = 0
        self.current_ratings_frame_idx = 0
        self.data_idx = 0
        self.ratings_frame = None

        self._max_history_length = history_length
        self._ignore_last_n: int = ignore_last_n
        self._cache = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self._sample_ratio: float = sample_ratio

        self.history_items_key = history_items_key
        self.candidate_items_key = candidate_items_key
        self.history_feature_columns = history_feature_columns
        self.candidate_feature_columns = candidate_feature_columns
        self.nonseq_columns = nonseq_columns
        self.history_feature_column_names = [k for k, _ in history_feature_columns.items()]
        self.candidate_feature_column_names = [k for k, _ in candidate_feature_columns.items()]
        self.nonseq_column_names = [k for k, _ in nonseq_columns.items()]
        self.itemid_column_name = itemid_column_name
        self.history_ratings_column_name = history_ratings_column_name
        self.candidate_ratings_column_name = candidate_ratings_column_name
        self.history_timestamps_column_name = history_timestamps_column_name
        self.candidate_timestamps_column_name = candidate_timestamps_column_name
        self.history_date_column_name = history_date_column_name
        self.candidate_date_column_name = candidate_date_column_name
        self.fill_dates = fill_dates

        extra_column_names = [
            self.candidate_ratings_column_name,
            self.history_timestamps_column_name,
            self.candidate_timestamps_column_name,
            self.history_date_column_name,
            self.candidate_date_column_name
        ]
        self.column_names = (
                self.history_feature_column_names +
                self.candidate_feature_column_names +
                self.nonseq_column_names +
                extra_column_names
        )

        self.inner_delim = inner_delim
        self.cut_off_time = cut_off_time
        self.is_train = is_train

        self.is_ads = is_ads
        self.sep = sep
        self.multi_csv_iterator = None
        self._max_candidate_num: int = num_rerank

        self.use_repadding = use_repadding
        if self.use_repadding and rank == 0:
            logging.info('Using sequence expand by repadding')

        self.token_per_item = token_per_item

        self.use_jagged_data = use_jagged_data
        self.use_dynamic_padding = use_dynamic_padding
        if use_dynamic_padding:
            logging.info(f'Rank {rank}: Using dynamic padding (batch-longest)')

        self.padding_side_origin = padding_side_origin
        self.padding_side = padding_side if padding_side else padding_side_origin

    def init_multi_csv_iterator(self) -> None:
        if not self.multi_csv_iterator:
            worker_info = torch.utils.data.get_worker_info()
            self.multi_csv_iterator = MultiFileIterator(self.files, self.sep, self.rank,
                                                        self.world_size, self.column_names,
                                                        self.inner_delim,
                                                        itemid_column_name=self.itemid_column_name,
                                                        file_format='parquet',
                                                        split_chunks_by_worker_id=False)

    def __len__(self):
        return DEFAULT_DATA_LEN

    def __iter__(self):
        if self.use_jagged_data:
            it = map(self.load_jagged_item, self.multi_csv_iterator)
        else:
            # 当前df的数据索引
            it = map(self.load_item, self.multi_csv_iterator)
        return it

    def _truncate_or_pad_seq(
            self,
            y: List[Union[int, List[int]]],
            target_len: int,
            chronological: bool,
            max_len_per_item: int = 1,
            padding_side: str = "right"
    ) -> List[Union[int, List[int]]]:
        """
        将序列 y 调整到长度为 target_len。如果 len_per_item > 0，则假定 y 中的每个元素本身是长度可变的列表，
        先对齐（truncate 或 pad）每个内层列表到长度 len_per_item，再对齐外层列表到长度 target_len；否则仅对齐外层列表。

        参数:
        - y: 原始序列，要么是 List[int]（len_per_item=0），要么是 List[List[int]]（len_per_item>0）。
        - target_len: 目标序列长度。外层长度会被截断或补齐到此值。
        - chronological: 当外层长度大于 target_len 时，决定保留头部还是尾部。
                        True 表示保留最靠近"当前"的末尾 target_len 个元素，False 表示保留最前面的 target_len 个元素。
        - max_len_per_item: 如果 >0，表示要先把 y 中每个元素（当作列表）对齐到长度 len_per_item，再再做外层对齐。
                        如果 =0，则认为 y 中元素是单个整数，直接对齐外层长度。
        - padding_side: padding方向，"left"在左侧padding，"right"在右侧padding。默认"right"。

        返回值:
        - List[Union[int, List[int]]]: 长度恰为 target_len 的序列；当 len_per_item>0 时，内层列表也固定为长度 len_per_item，
        padding 用 0 填充，截断时取头部（默认）。
        """
        # —— 1. 如果需要对齐内层列表，先对齐每个内层
        if max_len_per_item > 1:
            aligned_inner = []
            for elem in y:
                # 如果原始 elem 不是 list，也把它当作长度 1 的列表处理
                if not isinstance(elem, list):
                    current = [elem]
                else:
                    current = elem.copy()

                # 对齐到 len_per_item: 先截断或填 0
                if len(current) < max_len_per_item:
                    # 前置padding
                    current = [0] * (max_len_per_item - len(current)) + current
                elif len(current) > max_len_per_item:
                    # 截断：只保留最前面的 len_per_item 个元素
                    current = current[:max_len_per_item]

                aligned_inner.append(current)
            y = aligned_inner

        # —— 2. 外层长度对齐（truncate or pad）
        y_len = len(y)
        if y_len < target_len:
            if max_len_per_item > 1:
                pad_unit = [0] * max_len_per_item
                pads = [pad_unit for _ in range(target_len - y_len)]
            else:
                pads = [0] * (target_len - y_len)
            if padding_side == "left":
                y = pads + y
            else:
                y = y + pads

        elif y_len > target_len:
            if chronological:
                # 保留末尾 target_len 个元素
                y = y[-target_len:]
            else:
                # 保留最前面 target_len 个元素
                y = y[:target_len]

        if len(y) != target_len:
            raise ValueError(f"对齐后序列长度不等于 target_len（{target_len}），got {len(y)}")

        return y

    @staticmethod
    def _truncate_seq_only(
            y: List[Union[int, List[int]]],
            max_len: int,
            chronological: bool,
            max_len_per_item: int = 1
    ) -> List[Union[int, List[int]]]:
        """Only truncate, never pad. Used with dynamic padding where collate_fn handles padding."""
        # Inner alignment (same as _truncate_or_pad_seq - still needed for multi-value features)
        if max_len_per_item > 1:
            aligned_inner = []
            for elem in y:
                if not isinstance(elem, list):
                    current = [elem]
                else:
                    current = elem.copy()
                if len(current) < max_len_per_item:
                    current = [0] * (max_len_per_item - len(current)) + current
                elif len(current) > max_len_per_item:
                    current = current[:max_len_per_item]
                aligned_inner.append(current)
            y = aligned_inner

        # Outer: only truncate, no pad
        if len(y) > max_len:
            if chronological:
                y = y[-max_len:]
            else:
                y = y[:max_len]
        return y

    def load_item(self, data) -> Dict[str, torch.Tensor]:
        """
        加载单个数据项。

        :param data: 单个数据项。
        :return: 处理后的数据项，
        """

        def eval_as_list(x, ignore_last_n) -> List[int]:
            try:
                # 处理numpy数组
                if isinstance(x, np.ndarray):
                    if x.ndim == 0:
                        y_list = [x.item()]
                    elif x.dtype == object:
                        if len(x) == 0:
                            y_list = []
                        else:
                            # 每个内层数组都解析为二维列表的一行
                            y_list = []
                            for item in x:
                                if isinstance(item, np.ndarray):
                                    if item.ndim == 0:
                                        y_list.append([item.item()])  # 标量转为单元素列表
                                    else:
                                        y_list.append(item.tolist())  # 数组转为列表
                                else:
                                    y_list.append([item])  # 非数组元素也转为单元素列表
                    else:
                        y_list = x.tolist()
                # 处理字符串数据
                else:
                    if isinstance(x, str):
                        x = x.split(',')
                        if '.' in x[0]:
                            y_list = list(map(float, x))
                        else:
                            y_list = list(map(int, x))
                    else:
                        y_list = [x]

                # 时间顺序处理
                if not self._chronological:  # 如果需要逆序
                    y_list.reverse()

                # 忽略最近的n个数据
                if ignore_last_n > 0:
                    y_list = y_list[:-ignore_last_n] if len(y_list) > ignore_last_n else []
                    if not y_list:
                        y_list = [0]

                return y_list
            except AttributeError as e:
                logging.error("AttributeError: %s", str(e))
                raise e
            except ValueError as e:
                logging.error("ValueError: %s", str(e))
                raise e
            except Exception as e:
                logging.error("Other Exception: %s", str(e))
                raise e

        def eval_int_list(x, ignore_last_n: int, shift_id_by: int, sampling_kept_mask: Optional[List[bool]] = None) -> \
                Tuple[List[int], int]:
            y = eval_as_list(x, ignore_last_n=ignore_last_n)
            if sampling_kept_mask is not None:
                y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
            y_len = len(y)
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        ###############################################
        # 读取特征
        max_len_per_multival = {}  # 每个特征的最大长度，通过config读取，单值特征为1，多值特征为n
        # 读取history部分
        history_feature_list = {}
        history_feature_len = {}
        for column_name in self.history_feature_column_names:
            seq_feature, seq_len = eval_int_list(
                data[column_name], self._ignore_last_n, shift_id_by=self._shift_id_by
            )
            history_feature_list[column_name] = seq_feature
            history_feature_len[column_name] = seq_len
            feat_max_len = self.history_feature_columns.get(column_name).get("max_len", 1)
            max_len_per_multival[column_name] = feat_max_len

        history_timestamps, _ = eval_int_list(
            data[self.history_timestamps_column_name], self._ignore_last_n, 0)
        history_ratings, _ = eval_int_list(
            data[self.history_ratings_column_name], self._ignore_last_n, 0)
        history_dates, _ = eval_int_list(
            data[self.history_date_column_name], self._ignore_last_n, 0)

        # 读取candidate部分
        candidate_feature_list = {}
        candidate_feature_len = {}
        for column_name in self.candidate_feature_column_names:
            seq_feature, seq_len = eval_int_list(
                data[column_name], self._ignore_last_n, shift_id_by=self._shift_id_by)
            candidate_feature_list[column_name] = seq_feature
            candidate_feature_len[column_name] = seq_len
            feat_max_len = self.candidate_feature_columns.get(column_name).get("max_len", 1)
            max_len_per_multival[column_name] = feat_max_len

        candidate_ratings, candidate_ratings_len = eval_int_list(
            data[self.candidate_ratings_column_name], self._ignore_last_n, 0)
        candidate_timestamps, candidate_timestamps_len = eval_int_list(
            data[self.candidate_timestamps_column_name], self._ignore_last_n, 0)
        candidate_dates, _ = eval_int_list(
            data[self.candidate_date_column_name], self._ignore_last_n, 0)
        if self.fill_dates:
            candidate_dates = [candidate_dates[0]] * candidate_timestamps_len

        if candidate_timestamps_len != candidate_ratings_len:
            raise ValueError(
                f"timestamps len {candidate_timestamps_len} differs from ratings len {candidate_ratings_len}.")

        # 读取user部分
        non_seq_feature_list = {}
        non_seq_feature_len = {}
        for column_name in self.nonseq_column_names:
            seq_feature, seq_len = eval_int_list(
                data[column_name], self._ignore_last_n, shift_id_by=self._shift_id_by)
            non_seq_feature_list[column_name] = seq_feature
            non_seq_feature_len[column_name] = seq_len
            feat_max_len = self.nonseq_columns.get(column_name).get("max_len", 1)
            max_len_per_multival[column_name] = feat_max_len

        #################################################################
        # 对于candidate, 根据date分割train和test, 直接设置对应的掩码（现在要求cut_off_time为date格式，yymmdd）
        train_mask = [cand_date < self.cut_off_time for cand_date in candidate_dates]
        test_mask = [cand_date >= self.cut_off_time for cand_date in candidate_dates]
        history_mask = [hist_date < self.cut_off_time for hist_date in history_dates]

        if not self.is_train:  # test阶段
            candidate_timestamps = list(compress(candidate_timestamps, test_mask))
            candidate_ratings = list(compress(candidate_ratings, test_mask))
            candidate_dates = list(compress(candidate_dates, test_mask))
            for k, v in candidate_feature_list.items():
                v = list(compress(v, test_mask))
                candidate_feature_list[k] = v
        else:  # train阶段
            candidate_timestamps = list(compress(candidate_timestamps, train_mask))
            candidate_ratings = list(compress(candidate_ratings, train_mask))
            candidate_dates = list(compress(candidate_dates, train_mask))
            for k, v in candidate_feature_list.items():
                v = list(compress(v, train_mask))
                candidate_feature_list[k] = v

        ####################################################
        # 处理序列长度，padding
        def candidate_feature_interleave(candidate_feature: list, n: int = 2) -> list:
            length = len(candidate_feature)
            return [candidate_feature[i // n] for i in range(length * n)]

        # 0610数据，从左到右是从旧到新
        # 处理history部分
        # Dynamic padding模式下，先去掉预padding的0，算出真实有效长度
        if self.use_dynamic_padding:
            if self.padding_side_origin == "left":
                # 上游数据是左padding: [0,0,...,0, 真实数据...]
                # 从左向右找到第一个非0 timestamp，即为有效数据的起点
                _leading_zeros = 0
                for i, ts_val in enumerate(history_timestamps):
                    if ts_val != 0:
                        _leading_zeros = i
                        break
                else:
                    _leading_zeros = len(history_timestamps)  # 全0序列
                _real_hist_len = len(history_timestamps) - _leading_zeros
                history_length = min(_real_hist_len, self._max_history_length)
                # 截取有效部分（去掉左侧0-padding），保留最新的history_length个
                _trim_start = _leading_zeros + max(0, _real_hist_len - history_length)
                _trim_end = _leading_zeros + _real_hist_len
                if _trim_start > 0 or _trim_end < len(history_timestamps):
                    history_timestamps = history_timestamps[_trim_start:_trim_end]
                    history_ratings = history_ratings[_trim_start:_trim_end]
                    history_dates = history_dates[_trim_start:_trim_end]
                    for k in history_feature_list:
                        v = history_feature_list[k]
                        if isinstance(v, list) and len(v) > (len(v) - _leading_zeros):
                            history_feature_list[k] = v[_trim_start:_trim_end]
            else:
                # 上游数据是右padding: [真实数据..., 0,0,...,0]
                # 从右向左找到第一个非0 timestamp，即为有效数据的终点
                _trailing_zeros = 0
                for i in range(len(history_timestamps) - 1, -1, -1):
                    if history_timestamps[i] != 0:
                        _trailing_zeros = len(history_timestamps) - 1 - i
                        break
                else:
                    _trailing_zeros = len(history_timestamps)  # 全0序列
                _real_hist_len = len(history_timestamps) - _trailing_zeros
                history_length = min(_real_hist_len, self._max_history_length)
                # 截取有效部分（去掉右侧0-padding），保留最新的history_length个
                _trim_start = max(0, _real_hist_len - history_length)
                _trim_end = _real_hist_len
                if _trim_start > 0 or _trim_end < len(history_timestamps):
                    history_timestamps = history_timestamps[_trim_start:_trim_end]
                    history_ratings = history_ratings[_trim_start:_trim_end]
                    history_dates = history_dates[_trim_start:_trim_end]
                    for k in history_feature_list:
                        v = history_feature_list[k]
                        if isinstance(v, list) and len(v) > (len(v) - _trailing_zeros):
                            history_feature_list[k] = v[_trim_start:_trim_end]
        else:
            history_length = min(len(history_timestamps), self._max_history_length)

        # Dynamic padding: only truncate, collate_fn will pad to batch-max
        _truncate_fn = self._truncate_seq_only if self.use_dynamic_padding else self._truncate_or_pad_seq

        history_timestamps = _truncate_fn(history_timestamps, self._max_history_length, True)
        history_ratings = _truncate_fn(history_ratings, self._max_history_length, True)
        history_dates = _truncate_fn(history_dates, self._max_history_length, True)

        for k, v in history_feature_list.items():
            v = _truncate_fn(v, self._max_history_length, True, max_len_per_multival.get(k))
            history_feature_list[k] = v

        # 处理candidate部分
        max_candidate_num = self._max_candidate_num
        # Dynamic padding模式下，先去掉candidate的预padding
        if self.use_dynamic_padding:
            if self.padding_side_origin == "left":
                _cand_leading_zeros = 0
                for i, ts_val in enumerate(candidate_timestamps):
                    if ts_val != 0:
                        _cand_leading_zeros = i
                        break
                else:
                    _cand_leading_zeros = len(candidate_timestamps)  # 全0
                _real_cand_len = len(candidate_timestamps) - _cand_leading_zeros
                actual_candidate_num = min(_real_cand_len, max_candidate_num)
                _cand_trim_start = _cand_leading_zeros + max(0, _real_cand_len - actual_candidate_num)
                _cand_trim_end = _cand_leading_zeros + _real_cand_len
                if _cand_trim_start > 0 or _cand_trim_end < len(candidate_timestamps):
                    candidate_timestamps = candidate_timestamps[_cand_trim_start:_cand_trim_end]
                    candidate_ratings = candidate_ratings[_cand_trim_start:_cand_trim_end]
                    candidate_dates = candidate_dates[_cand_trim_start:_cand_trim_end]
                    for k in candidate_feature_list:
                        v = candidate_feature_list[k]
                        if isinstance(v, list) and len(v) > (len(v) - _cand_leading_zeros):
                            candidate_feature_list[k] = v[_cand_trim_start:_cand_trim_end]
            else:
                _cand_trailing_zeros = 0
                for i in range(len(candidate_timestamps) - 1, -1, -1):
                    if candidate_timestamps[i] != 0:
                        _cand_trailing_zeros = len(candidate_timestamps) - 1 - i
                        break
                else:
                    _cand_trailing_zeros = len(candidate_timestamps)  # 全0
                _real_cand_len = len(candidate_timestamps) - _cand_trailing_zeros
                actual_candidate_num = min(_real_cand_len, max_candidate_num)
                _cand_trim_start = max(0, _real_cand_len - actual_candidate_num)
                _cand_trim_end = _real_cand_len
                if _cand_trim_start > 0 or _cand_trim_end < len(candidate_timestamps):
                    candidate_timestamps = candidate_timestamps[_cand_trim_start:_cand_trim_end]
                    candidate_ratings = candidate_ratings[_cand_trim_start:_cand_trim_end]
                    candidate_dates = candidate_dates[_cand_trim_start:_cand_trim_end]
                    for k in candidate_feature_list:
                        v = candidate_feature_list[k]
                        if isinstance(v, list) and len(v) > (len(v) - _cand_trailing_zeros):
                            candidate_feature_list[k] = v[_cand_trim_start:_cand_trim_end]
        else:
            actual_candidate_num = min(len(candidate_timestamps), max_candidate_num)
        candidate_ratings = _truncate_fn(candidate_ratings, max_candidate_num, True)
        candidate_ratings = candidate_feature_interleave(candidate_ratings, self.token_per_item)
        candidate_timestamps = _truncate_fn(candidate_timestamps, max_candidate_num, True)
        candidate_dates = _truncate_fn(candidate_dates, max_candidate_num, True)
        if self.token_per_item == 2:
            candidate_timestamps = candidate_timestamps + candidate_timestamps
            candidate_dates = candidate_dates + candidate_dates

        for k, v in candidate_feature_list.items():
            v = _truncate_fn(v, self._max_candidate_num, True, max_len_per_multival.get(k))
            v = candidate_feature_interleave(v, self.token_per_item)
            candidate_feature_list[k] = v

        # 处理user部分 (always length 1, no need for dynamic)
        for k, v in non_seq_feature_list.items():
            v = self._truncate_or_pad_seq(v, 1, True, max_len_per_multival.get(k), padding_side=self.padding_side)
            v = candidate_feature_interleave(v, self.token_per_item)
            candidate_feature_list[k] = v

        #######################################################
        # 处理训练数据标签
        labels = candidate_ratings  # 新数据方案candidate_action_type本身只有0曝光1下载，直接可以用做ground_truth

        # 处理模型权重
        loss_weights = [1 if t != 0 else 0 for t in candidate_timestamps]

        history_ids = history_feature_list.get(self.history_items_key)
        candidate_ids = candidate_feature_list.get(self.candidate_items_key)

        # 间隔处loss weights为0，但是rating需要在embedding内
        if self.use_repadding:  # 间隔处是啥意思？没看懂: 猜测是补零位
            candidate_ratings = [0 if x == -1 else x for x in candidate_ratings]

        # aid为64位16进制数字，为了能让torch.int64不溢出，取其前15位
        try:
            # 确保 data['user_id'] 被正确读取,将 user_id 转换为字符串（如果它不是字符串）
            user_id = str(data['did'])
            # 去掉字符串中的所有非合法16进制字符
            clean_user_id = ''.join([c for c in user_id if c in '0123456789abcdefABCDEF'])
            # 取前15个字符并转换为整数
            uid = int(clean_user_id[0:min(15, len(clean_user_id))], 16)

        except Exception as e:
            # 如果发生错误，打印调试信息并设置默认值或抛出异常
            uid = 0  # 或者根据需求设置其他默认值

        ret = {
            "uid": torch.tensor(uid, dtype=torch.int64),
            "history_lengths": history_length,
            "candidate_lengths": actual_candidate_num,
            "history_action_type": torch.tensor(history_ratings, dtype=torch.int64),
            "candidate_action_type": torch.tensor(candidate_ratings, dtype=torch.int64),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "history_ids": torch.tensor(history_ids, dtype=torch.int64),
            "candidate_ids": torch.tensor(candidate_ids, dtype=torch.int64),
            self.history_timestamps_column_name: torch.tensor(history_timestamps, dtype=torch.int64),
            self.candidate_timestamps_column_name: torch.tensor(candidate_timestamps, dtype=torch.int64),
            self.history_date_column_name: torch.tensor(history_dates, dtype=torch.int64),
            self.candidate_date_column_name: torch.tensor(candidate_dates, dtype=torch.int64),
            self.candidate_ratings_column_name: torch.tensor(candidate_ratings, dtype=torch.int64)
        }

        for k, v in non_seq_feature_list.items():
            ret[k] = torch.tensor(v, dtype=torch.int64)
        for k, v in history_feature_list.items():
            ret[k] = torch.tensor(v, dtype=torch.int64)
        for k, v in candidate_feature_list.items():
            ret[k] = torch.tensor(v, dtype=torch.int64)

        return ret
