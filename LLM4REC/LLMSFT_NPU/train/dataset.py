from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CLDataset(Dataset):
    def __init__(self, data_file):
        super(CLDataset, self).__init__()
        self.data_path = data_file
        self.df = pd.read_csv(data_file)

    def __getitem__(self, idx):
        index_x, item_id_x, info_x, cluster_id_x, index_y, item_id_y, info_y, cluster_id_y = self.df.iloc[idx].tolist()
        index = [int(index_x), int(index_y)]
        info = [info_x, info_y]
        return index, info

    def __len__(self):
        return len(self.df)


@dataclass
class CLCollator:
    def __init__(self, base_model, cache_dir, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True,
                                                       cache_dir=cache_dir)
        self.max_length = max_length
        self.tokenizer.padding_side = "left"

    def __call__(self, batch) -> dict:
        index, item_info = zip(*batch)

        item_ids, describs = [], []
        for ids, info in zip(index, item_info):
            item_ids.extend(ids)
            describs.extend(info)

        token_ids, attention, token_type_ids = self.tokenizer.batch_encode_plus(describs,
                                                                                truncation=True,
                                                                                padding=True,
                                                                                max_length=self.max_length,
                                                                                return_tensors='pt',
                                                                                add_special_tokens=False).values()

        ones = torch.ones(len(item_ids), 1).long()
        attention_mask = torch.cat((attention, ones), dim=1)
        item_ids = torch.LongTensor(item_ids)

        return {
            "inputs": token_ids,
            "inputs_mask": attention_mask,
            "item_ids": item_ids
        }


class PredictDataset(Dataset):
    def __init__(self, data_file):
        super(PredictDataset, self).__init__()
        self.data_path = data_file
        self.df = pd.read_csv(data_file)

    def __getitem__(self, idx):
        index, item_id, info, cluster_id = self.df.iloc[idx].tolist()
        return int(index), item_id, info

    def __len__(self):
        return len(self.df)


@dataclass
class PredictCollator:
    def __init__(self, base_model, cache_dir, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True,
                                                       cache_dir=cache_dir)
        self.max_length = max_length
        self.tokenizer.padding_side = "left"

    def __call__(self, batch) -> dict:
        index_ids, item_ids, describs = zip(*batch)
        token_ids, attention, token_type_ids = self.tokenizer.batch_encode_plus(describs,
                                                                                truncation=True,
                                                                                padding=True,
                                                                                max_length=self.max_length,
                                                                                return_tensors='pt',
                                                                                add_special_tokens=False).values()

        ones = torch.ones(len(index_ids), 1).long()
        attention_mask = torch.cat((attention, ones), dim=1)
        index_ids = torch.LongTensor(index_ids)

        return {
            "inputs": token_ids,
            "inputs_mask": attention_mask,
            "index_ids": index_ids,
            "item_ids": item_ids
        }
