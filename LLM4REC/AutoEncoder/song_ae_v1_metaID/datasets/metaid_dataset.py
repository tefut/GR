import torch
from torch.utils.data import Dataset


class MetaIDDataset(Dataset):
    def __init__(
        self,
        input_df,
        feat_map,
        device,
    ):
        super().__init__()
        self.raw_df = input_df
        self.feat_map = feat_map
        self.device = device

    def __len__(self):
        return self.raw_df.shape[0]

    def __getitem__(self, idx):
        row = self.raw_df.iloc[idx]
        row_as_dict = dict()
        for k in self.feat_map.keys():
            row_as_dict[k] = torch.tensor(self.feat_map[k][row[k]]).to(self.device)
        return row_as_dict
