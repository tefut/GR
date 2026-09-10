import os
import os.path as osp
import collections

import pandas as pd


def metaid_v1_data_process(ds_root, sel_cols, all_col_nm=False, file_type='orc', sep='|', tag_spliter=None):

    if (not isinstance(sel_cols, list)) or len(sel_cols) <= 0:
        sel_cols = ['song_id']
    if (not isinstance(all_col_nm, list)) or len(all_col_nm) <= 0:
        all_col_nm = ['song_id']

    raw_data_lst = list()
    for data_file in os.listdir(ds_root):
        if file_type == 'orc':
            curr_df = pd.read_orc(osp.join(ds_root, data_file))
        elif file_type == 'csv':
            curr_df = pd.read_csv(osp.join(ds_root, data_file), sep=sep, names=all_col_nm)
        else:
            raise NotImplementedError
        raw_data_lst.append(curr_df)
    raw_data_lst = pd.concat(raw_data_lst)

    proc_df = raw_data_lst[sel_cols]
    if tag_spliter:
        proc_df = proc_df.apply(lambda x: x.str.split(tag_spliter, n=1).str.get(0), axis=1)
    proc_df.dropna(axis='index', inplace=True)
    
    feats_dict = dict()
    for _ in proc_df.columns:
        unq_vals = pd.unique(proc_df.loc[:, _])
        feat_dict = collections.defaultdict(int)
        for idx, nm in enumerate(unq_vals):
            feat_dict[nm] = idx
        feats_dict[_] = feat_dict

    return proc_df, feats_dict
