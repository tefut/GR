# -*- coding: utf-8 -*-
"""
Embedding数据集类
支持配置化列名映射
支持npz格式、Parquet流式格式和CSV格式
"""

import logging
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from SID_Gen.utils.column_mapping import ColumnMapper, create_column_mapper

logger = logging.getLogger(__name__)


class EmbDataset(data.Dataset):
    def __init__(
            self,
            data_path: str,
            column_mapper: Optional[Dict[str, Any]] = None,
            norm: bool = False,
    ):
        """
        初始化Embedding数据集
        
        Args:
            data_path: npz文件路径
            column_mapper: 列名映射器
            norm: 是否对输入embedding做L2归一化
        """
        self.data_path = data_path
        self.column_mapper = column_mapper
        self.norm = norm

        load_data = np.load(data_path, allow_pickle=True)  # *.npz

        pk_col = self.column_mapper.get("primary_key", "id")
        emb_col = self.column_mapper.get("embedding_name", "embedding")

        if pk_col in load_data:
            self.item_ids = load_data[pk_col]
        elif "app_id" in load_data:
            self.item_ids = load_data["app_id"]
        else:
            self.item_ids = np.array([f"item_{i}" for i in range(len(load_data[emb_col]))])

        self.embeddings = load_data[emb_col].astype(np.float32, copy=False)

        nan_mask = np.isnan(self.embeddings)
        if nan_mask.any():
            self.embeddings[nan_mask] = 0.0

        inf_mask = np.isinf(self.embeddings)
        if inf_mask.any():
            self.embeddings[inf_mask] = 0.0

        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        item_id = self.item_ids[index]

        # 统一转 str
        if isinstance(item_id, (bytes, np.bytes_)):
            item_id = item_id.decode("utf-8", errors="ignore")
        else:
            item_id = str(item_id)

        emb = self.embeddings[index]
        if self.norm:
            norms = np.linalg.norm(emb)
            if norms > 0:
                emb = emb / norms
        tensor_emb = torch.from_numpy(emb)

        return item_id, tensor_emb

    def __len__(self):
        return len(self.embeddings)


class CsvEmbDataset(data.Dataset):
    def __init__(self, item_ids: List[str], embeddings: np.ndarray, norm: bool = False):
        self.item_ids = item_ids
        self.embeddings = embeddings
        self.dim = self.embeddings.shape[-1]
        self.norm = norm

    def __getitem__(self, index):
        item_id = self.item_ids[index]
        emb = self.embeddings[index]
        if self.norm:
            norms = np.linalg.norm(emb)
            if norms > 0:
                emb = emb / norms
        tensor_emb = torch.from_numpy(emb)
        return item_id, tensor_emb

    def __len__(self):
        return len(self.embeddings)


def create_emb_dataset(
        data_type: str = "npz",
        data_path: str = "",
        data_dir: str = "",
        manifest_path: str = "",
        columns: Optional[List[str]] = None,
        column_mapper: Optional[Dict[str, Any]] = None,
        max_cache_shards: int = 4,
        shuffle_shards: bool = False,
        prefetch_shards: int = 0,
        worker_id: Optional[int] = None,
        seed: int = 42,
        norm: bool = False,
        csv_sep: str = ",",
        expected_emb_dim: Optional[int] = None,
) -> data.Dataset:
    """
    创建Embedding数据集的工厂函数
    
    根据data_type自动选择EmbDataset或EmbStreamingDataset
    
    Args:
        data_type: 数据格式类型，"npz" 或 "parquet"
        data_path: npz文件路径（用于data_type="npz"）
        data_dir: parquet数据目录（用于data_type="parquet"）
        manifest_path: manifest文件路径（用于data_type="parquet"）
        columns: 需要读取的列名列表（用于data_type="parquet"）
        column_mapper: 列名映射器
        max_cache_shards: 最大缓存分片数（用于data_type="parquet"）
        shuffle_shards: 是否打乱分片顺序（用于data_type="parquet"）
        prefetch_shards: 预取分片数（用于data_type="parquet"）
        worker_id: worker ID（用于data_type="parquet"）
        seed: 随机种子（用于data_type="parquet"）
        norm: 是否对输入embedding做L2归一化
        csv_sep: CSV文件分隔符（用于data_type="csv"）
        expected_emb_dim: 期望的embedding维度，为None时不做过滤；
                          设置后丢弃维度不匹配的样本（用于清除CSV脏数据）
        
    Returns:
        Dataset对象
    """
    if data_type == "parquet":
        # 使用流式Parquet数据集
        if not manifest_path:
            raise ValueError(
                "manifest_path is required when data_type='parquet'. "
                "Use generate_manifest.py to create a manifest file."
            )

        # 动态导入避免不必要的依赖
        from SID_Gen.my_datasets.streaming_dataset import EmbStreamingDataset

        dataset = EmbStreamingDataset(
            manifest_path=manifest_path,
            columns=columns,
            max_cache_shards=max_cache_shards,
            shuffle_shards=shuffle_shards,
            prefetch_shards=prefetch_shards,
            worker_id=worker_id,
            seed=seed,
        )

        logger.info(
            "Created EmbStreamingDataset: total=%d, shards=%d",
            len(dataset), len(dataset.manifest.shards)
        )

        return dataset

    elif data_type == "npz":
        if not data_path:
            raise ValueError("data_path is required when data_type='npz'")

        dataset = EmbDataset(data_path=data_path, column_mapper=column_mapper, norm=norm)

        logger.info("Created EmbDataset: total=%d", len(dataset))

        return dataset

    elif data_type == "csv":
        if not data_path:
            raise ValueError("data_path is required when data_type='csv'")

        pk_col = column_mapper.get("primary_key", "id") if column_mapper else "id"
        emb_col = column_mapper.get("embedding_name", "embedding") if column_mapper else "embedding"

        df = pd.read_csv(data_path, sep=csv_sep)

        if pk_col in df.columns:
            item_ids = df[pk_col].astype(str).tolist()
        else:
            item_ids = [f"item_{i}" for i in range(len(df))]

        emb_series = df[emb_col].apply(lambda x: eval(x) if isinstance(x, str) else x)

        if expected_emb_dim is not None:
            total_rows = len(emb_series)
            valid_mask = emb_series.apply(
                lambda x: isinstance(x, (tuple, list, np.ndarray)) and len(x) == expected_emb_dim
            )
            filtered_count = total_rows - valid_mask.sum()
            if filtered_count > 0:
                logger.warning(
                    "过滤掉 %d 条维度不为 %d 的脏数据（总行数: %d）",
                    filtered_count, expected_emb_dim, total_rows,
                )
                emb_series = emb_series[valid_mask]
                item_ids = [item_ids[i] for i, v in enumerate(valid_mask) if v]

        embeddings = np.array(emb_series.tolist(), dtype=np.float32)

        nan_mask = np.isnan(embeddings)
        if nan_mask.any():
            embeddings[nan_mask] = 0.0

        inf_mask = np.isinf(embeddings)
        if inf_mask.any():
            embeddings[inf_mask] = 0.0

        dataset = CsvEmbDataset(item_ids, embeddings, norm=norm)

        logger.info("Created CsvEmbDataset: total=%d", len(dataset))

        return dataset

    else:
        raise ValueError(
            f"Unsupported data_type: {data_type}. "
            f"Supported types: 'npz', 'parquet', 'csv'"
        )
