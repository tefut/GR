import os
import time
import random
from typing import List, Dict, Any
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pyarrow.orc as orc
import pyarrow.parquet as pq


class GRDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 exclude_features: List[str],
                 ordered_features_names: List[str] = None,
                 extensions: List[str] = None,
                 chunk_size: int = 100,
                 shuffle_file: bool = True,
                 shuffle_line: bool = True,
                 seed: int = 2025,
                 is_train: bool = True):
        self.data_dir = data_dir
        self.exclude_features = exclude_features
        self.ordered_features_names = ordered_features_names
        self.extensions = extensions or ['txt', 'csv', 'orc', 'parquet']
        self.chunk_size = chunk_size
        self.shuffle_file = shuffle_file
        self.shuffle_line = shuffle_line
        self.seed = seed
        self.is_train = is_train

        # 查找符合条件的文件
        self.file_list = self._find_files()
        if not self.file_list:
            raise RuntimeError(f"在 {data_dir} 目录下未找到扩展名是 {extensions} 的文件")

        # 计算每个文件的块数
        compute_chunk_start_time = time.time()
        self.file_chunk_counts = self._compute_file_chunks()
        compute_chunk_end_time = time.time()
        print(f"the time duration of compute file chunks: {compute_chunk_end_time - compute_chunk_start_time}")

        # 创建文件块索引表（file_ids, chunk_idx_in_file）
        self.chunk_indices = self._create_chunk_indices()
        print("length of chunk indices", len(self.chunk_indices))

        self.reset_seed()

    def _find_files(self) -> List[str]:
        """查找所有符合条件的文件"""
        file_list = []
        extensions = [ext.lower() for ext in self.extensions]

        for root, _, fnames in sorted(os.walk(self.data_dir)):
            for fname in sorted(fnames):
                if any(fname.lower().endswith(ext) for ext in extensions):
                    file_list.append(os.path.join(root, fname))

        return file_list

    def _compute_file_chunks(self) -> Dict[str, int]:
        """计算每个文件的块数"""
        file_chunk_counts = {}
        for _, file_path in enumerate(self.file_list):
            if file_path.endswith("orc"):
                data = orc.ORCFile(file_path)
                num_lines = data.nrows
                file_chunk_counts[file_path] = (num_lines + self.chunk_size - 1) // self.chunk_size
                del data
            elif any(file_path.endswith(ext) for ext in ['.csv', '.tsv']):
                with open(file_path, 'r') as fin:
                    num_lines = sum(1 for _ in fin)
                    file_chunk_counts[file_path] = (num_lines + self.chunk_size - 1) // self.chunk_size
            elif file_path.endswith("parquet"):
                data = pq.read_table(file_path)
                num_lines = data.num_rows
                file_chunk_counts[file_path] = (num_lines + self.chunk_size - 1) // self.chunk_size
            else:
                file_chunk_counts[file_path] = 1

            print(f"{file_path} num_lines: {num_lines}")
        return file_chunk_counts

    def _create_chunk_indices(self) -> List[Dict[str, int]]:
        """创建块索引表"""
        chunk_indices = []
        for file_idx, file_path in enumerate(self.file_list):
            num_chunks = self.file_chunk_counts.get(file_path)
            for chunk_in_file_idx in range(num_chunks):
                chunk_indices.append({
                    "file_idx": file_idx,
                    "chunk_in_file_idx": chunk_in_file_idx
                })

        # 打乱块顺序
        if self.shuffle_file:
            random.shuffle(chunk_indices)
        return chunk_indices

    def set_distributed(self, world_size, rank):
        mod = len(self.chunk_indices) % world_size
        if mod != 0:
            new_length = len(self.chunk_indices) - mod
            self.chunk_indices = self.chunk_indices[:-new_length]
            print(f"length of chunk_indices: {new_length}, drop num batch: {mod}")
        self.chunk_indices = self.chunk_indices[rank::world_size]
        print(f"process {rank}/{world_size} num of chunk_indices: {len(self.chunk_indices)}")

    def _default_loader(self, file_path: str) -> pd.DataFrame:
        """默认文件加载器"""
        if file_path.endswith("csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith("orc"):
            orc_file = orc.ORCFile(file_path)
            return orc_file.read().to_pandas()
        elif file_path.endswith("parquet"):
            return pq.read_table(file_path).to_pandas()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return pd.DataFrame({"text": lines})

    def _loader_data(self, file_path, chunk_in_file_idx):
        if file_path.endswith("csv") or file_path.endswith("orc"):
            if file_path.endswith("orc"):
                orc_file = orc.ORCFile(file_path)
                df = orc_file.read().to_pandas()
            else:
                df = pd.read_csv(file_path)
            exclude_features_list = [feature for feature in self.exclude_features if feature in df.columns]
            df.drop(columns=exclude_features_list, inplace=True)

            start_idx = chunk_in_file_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, df.nrows)

            if end_idx <= start_idx:
                chunk_data = []
            else:
                chunk_data = df.iloc[start_idx: end_idx]
            return chunk_data
        elif file_path.endswith("parquet"):
            table = pq.read_table(file_path)
            start_idx = chunk_in_file_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, table.num_rows)
            if end_idx <= start_idx:
                chunk_data = []
            else:
                mask = [i > chunk_in_file_idx * self.chunk_size and i <= end_idx for i in range(table.num_rows)]
                chunk_data = table.filter(mask).to_pandas()
            return chunk_data
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return pd.DataFrame({"text": lines})

    def reset_seed(self):
        """重置随机种子"""
        self.random_state = random.Random(self.seed)

    def __len__(self) -> int:
        """返回数据集的块数"""
        return len(self.chunk_indices)

    def __getitem__(self, idx: int) -> List[Any]:
        """获取一个数据块"""

        if torch.is_tensor(idx):
            idx = idx.tolist()

        chunk_info = self.chunk_indices[idx]
        file_idx = chunk_info["file_idx"]
        chunk_in_file_idx = chunk_info["chunk_in_file_idx"]
        file_path = self.file_list[file_idx]

        # 获取数据块
        chunk_data = self._loader_data(file_path, chunk_in_file_idx)
        return chunk_data


def collate_fn(chunks):
    batch = pd.concat(chunks)

    dict_data = batch.to_dict(orient='list')

    batch_data = {}
    for k, v in dict_data.items():
        mid_data = np.stack(v)
        if mid_data.shape[-1] == 1:
            data = np.squeeze(mid_data, axis=-1)
        else:
            data = mid_data
        if np.issubdtype(data.dtype, np.str_):
            batch_data[k] = data
        else:
            batch_data[k] = torch.tensor(data)
    return batch_data
