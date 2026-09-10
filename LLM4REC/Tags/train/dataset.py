from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from utils import build_prompt


class IMCIDataset(Dataset):
    def __init__(self, data):
        super(IMCIDataset, self).__init__()
        self.model_input = []
        for item_info in data:
            prompt, generate_input, _, _ = build_prompt(item_info)
            self.model_input.append(prompt)

    def __getitem__(self, index):
        return self.model_input[int(index)]

    def __len__(self):
        return len(self.model_input)


class IMCICollator:
    def __init__(self, base_model, cache_dir, max_length, ignore_index, device):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, cache_dir=cache_dir)

        self.max_length = max_length
        self.tokenizer.padding_side = "left"

        self.pad_token_id = len(self.tokenizer) - 1
        self.ignore_index = ignore_index
        self.device = device

    def __call__(self, batch):
        batch_max_length = max(len(item) + 1 for item in batch)

        inputs_list, targets_list = [], []

        for item in batch:
            item = self.tokenizer.encode(item)
            new_item = item.copy()

            new_item += [self.pad_token_id]

            padded = (new_item + [self.pad_token_id] * (batch_max_length - len(new_item)))
            inputs = torch.tensor(padded[:-1])
            targets = torch.tensor(padded[1:])

            mask = targets == self.pad_token_id
            indices = torch.nonzero(mask).squeeze()
            if indices.numel() > 1:
                targets[indices[1:]] = self.ignore_index

            if self.max_length is not None:
                inputs = inputs[:self.max_length]
                targets = targets[:self.max_length]

            inputs_list.append(inputs)
            targets_list.append(targets)

        inputs_tensor = torch.stack(inputs_list).to(self.device)
        targets_tensor = torch.stack(targets_list).to(self.device)

        return inputs_tensor, targets_tensor


class IMCIEDataset(Dataset):
    def __init__(self, content_data, cf_data):
        super(IMCIDataset, self).__init__()
        self.model_input = {}
        for item_info in content_data:
            prompt, generate_input, item_id, prompt_detail = build_prompt(item_info)
            self.model_input[item_id] = prompt, prompt_detail

        self.data_path = cf_data
        self.df = pd.read_csv(cf_data)

    def __getitem__(self, idx):
        index_x, item_id_x, info_x, cluster_id_x, index_y, item_id_y, info_y, cluster_id_y = self.df.iloc[idx].tolist()
        index = [int(index_x), int(index_y)]
        gen_info_x = self.model_input[item_id_x][1] if item_id_x in self.model_input else ""
        gen_info_y = self.model_input[item_id_y][1] if item_id_y in self.model_input else ""
        info_detail_x = self.model_input[item_id_x][3] if item_id_x in self.model_input else ""
        info_detail_y = self.model_input[item_id_y][3] if item_id_y in self.model_input else ""
        detail_info = [[info_detail_x[0], info_detail_x[1], info_detail_x[2]], \
                       [info_detail_y[0], info_detail_y[1], info_detail_y[2]]]
        gen_info = [gen_info_x, gen_info_y]
        emb_idx = [info_detail_x[3], info_detail_y[3]]
        return index, gen_info, detail_info, emb_idx

    def __len__(self):
        return len(self.df)


class IMCIECollator:
    def __init__(self, base_model, cache_dir, max_length, ignore_index, device):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, cache_dir=cache_dir)

        self.max_length = max_length
        self.tokenizer.padding_side = "left"

        self.pad_token_id = len(self.tokenizer) - 1
        self.ignore_index = ignore_index
        self.device = device

    def __call__(self, batch):
        index, gen_info, detail_info, emb_idx = zip(*batch)

        item_ids, describs = [], []
        token_ids_list, attention_mask_list = [], []
        for ids, info in zip(index, detail_info):
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

            token_ids_list.append(token_ids)
            attention_mask_list.append(attention_mask)

        cf_info = {
            "inputs": torch.cat(token_ids_list),
            "inputs_mask": torch.cat(attention_mask_list),
            "item_ids": item_ids
        }

        return cf_info


class IMCIEPredictDataset(Dataset):
    def __init__(self, content_data, data_file):
        super(IMCIEPredictDataset, self).__init__()
        self.data_path = data_file
        self.df = pd.read_csv(data_file)

        self.model_input = {}
        for item_info in content_data:
            prompt, generate_input, item_id = build_prompt(item_info)
            self.model_input[item_id] = prompt

    def __getitem__(self, idx):
        index, item_id, info, cluster_id = self.df.iloc[idx].tolist()
        info = self.model_input[item_id] if item_id in self.model_input else ""
        return int(index), item_id, info

    def __len__(self):
        return len(self.df)


@dataclass
class IMCIEPredictCollator:
    def __init__(self, base_model, cache_dir, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True,
                                                       cache_dir=cache_dir)
        self.max_length = max_length
        self.tokenizer.padding_side = "left"

    def __call__(self, batch) -> dict:
        item_encode_id, item_ids, describs = zip(*batch)
        token_ids, attention, token_type_ids = self.tokenizer.batch_encode_plus(describs,
                                                                                truncation=True,
                                                                                padding=True,
                                                                                max_length=self.max_length,
                                                                                return_tensors='pt',
                                                                                add_special_tokens=False).values()

        ones = torch.ones(len(item_ids), 1).long()
        attention_mask = torch.cat((attention, ones), dim=1)
        indexs = torch.LongTensor(item_ids)

        return {
            "inputs": token_ids,
            "inputs_mask": attention_mask,
            "item_encode_id": item_encode_id,
            "item_ids": item_ids
        }
