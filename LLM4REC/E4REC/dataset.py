import os
import pandas as pd
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


class SequentialDataset(Dataset):
    def __init__(self, data_path, maxlen):
        super(SequentialDataset, self).__init__()
        self.data_path = data_path
        self.maxlen = maxlen

        self.trainData, self.valData, self.testData = [], {}, {}
        self.allPos = {}

        for file in os.listdir(self.data_path):
            if file == 'part_0':
                data_pd = pd.read_csv(os.path.join(self.data_path, file), sep="|", engine='python',
                                      names=['user_id', 'user_seq'])
                for user, seq in zip(data_pd["user_id"], data_pd["user_seq"]):
                    items = [int(item) for item in seq.split("|")]
                    if len(items) >= 3:
                        items = items[(len(items) - self.maxlen):] if len(items) > self.maxlen else items
                        train_items = items[:-2]
                        length = len(train_items)
                        for t in range(length):
                            self.trainData.append([train_items[:-length + t], train_items[-length + t]])
                        self.testData[user] = [items[:-1], items[-1]]
                    else:
                        for t in range(len(items)):
                            self.trainData.append([items[:-len(items) + t], items[-len(items) + t]])
                        self.valData[user] = []
                        self.testData[user] = []
            elif file == 'part_0_sample':
                with open(os.path.join(self.data_path, file), 'r') as f:
                    for line in f:
                        line = line.strip().split('|')
                        user, items = line[0], [int(item) for item in line[1:]]
                        self.allPos[user] = items

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append(self.allPos[user] + self.testData[user])
        return posItems

    def __getitem__(self, idx):
        seq, label = self.trainData[idx]
        return seq, label

    def __len__(self):
        return len(self.trainData)


@dataclass
class SequentialCollator:
    def __call__(self, batch) -> dict:
        seqs, labels = zip(*batch)
        max_len = max(max([len(seq) for seq in seqs]), 2)
        inputs = [[0] * (max_len - len(seq)) + seq for seq in seqs]
        inputs_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs]
        labels = [[label] for label in labels]
        inputs, inputs_mask, labels = torch.LongTensor(inputs), torch.LongTensor(inputs_mask), torch.LongTensor(labels)

        return {
            "inputs": inputs,
            "inputs_mask": inputs_mask,
            "labels": labels
        }


class PredictDataset(Dataset):
    def __init__(self, data_path, maxlen):
        super(PredictDataset, self).__init__()
        self.data_path = data_path
        self.maxlen = maxlen
        self.data = []

        for file in os.listdir(self.data_path):
            if file.endswith("0_test") and not file.endswith("sample"):
                data_pd = pd.read_csv(os.path.join(self.data_path, file), sep="|", engine='python',
                                      names=['user_id', 'user_seq'])
                for user, seq in zip(data_pd["user_id"], data_pd["user_seq"]):
                    items = [int(item) for item in seq.split("|")]
                    length = min(len(items), self.maxlen)
                    items = items[(len(items) - length - 1):]
                    self.data.append([user, items])
        self.data = self.data

    def __getitem__(self, idx):
        uid, items = self.data[idx]
        return uid, items

    def __len__(self):
        return len(self.data)


@dataclass
class PredictCollator:
    def __call__(self, batch) -> dict:
        uids, items = zip(*batch)
        seqs = items
        max_len = max(max([len(seq) for seq in seqs]), 2)
        inputs = [[0] * (max_len - len(seq)) + seq for seq in seqs]
        inputs_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs]
        uids = [uid for uid in uids]
        inputs, inputs_mask = torch.LongTensor(inputs), torch.LongTensor(inputs_mask)

        return {
            "inputs": inputs,
            "inputs_mask": inputs_mask,
            "uids": uids
        }
