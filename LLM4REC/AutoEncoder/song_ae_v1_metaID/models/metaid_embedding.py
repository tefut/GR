from torch import nn


class MetaIDEmbd(nn.Module):
    def __init__(self, feats_dict, embd_dim=8):
        super(MetaIDEmbd, self).__init__()
        self.feats_dict = feats_dict
        self.embd_dim = embd_dim
        self.embd_dict = nn.ModuleDict()
        for k in self.feats_dict.keys():
            curr_mapping = self.feats_dict[k]
            self.embd_dict[k] = nn.Embedding(len(curr_mapping), self.embd_dim)

    def forward(self, x):
        ret_lst = list()
        for k in x.keys():
            ret_lst.append(self.embd_dict[k](x[k]))
        return ret_lst
