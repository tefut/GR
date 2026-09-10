import argparse
import os
import pickle
import random
import time
from collections import defaultdict
from io import open

import numpy as np
import pandas as pd
import torch

np.random.seed(12345)


def load_song_embeddings(item_embedding_file):
    item_embeddings_pd = pd.read_csv(item_embedding_file, sep="|", engine='python', names=['song_id', 'song_embedding'])
    print(f"the total number of item_embeddings_pd is: {len(item_embeddings_pd)}")
    print(item_embeddings_pd.head(5), "\n")
    return item_embeddings_pd


def concat_user_pd(file_path: str):
    df_list = []
    for file in os.listdir(file_path):
        if not file.startswith("part"):
            continue
        user_file = os.path.join(file_path, file)
        user_pd = pd.read_orc(user_file)
        df_list.append(user_pd)
    total_df = pd.concat(df_list, ignore_index=True)
    print(f"the total number of origin user_df is: {len(total_df)}")
    print(total_df.head(5), "\n")
    return total_df


def filter_embedding_by_freq(user_file_path: str, item_embedding_file: str, max_item_num: int = 200000):
    # 载入用户行为数据dataframe
    user_df = concat_user_pd(user_file_path)

    # 统计用户行为数据中，每个item出现的频率
    item_count = user_df["positive_song_seq"].str.split("#", expand=True).stack()
    item_count = item_count.value_counts()
    item_freq_pd = item_count.to_frame(name="count").reset_index()
    print("the number of item_count", len(item_count), "\n")
    print(item_freq_pd.head(5), "\n")

    # 基于item出现频率，对song embeddings进行过滤，保留前max_item_num个item，并对item重建索引
    item_embed_df = load_song_embeddings(item_embedding_file)
    filter_item_embed_df = item_freq_pd.merge(item_embed_df, left_on="index", right_on="song_id", how="inner")
    filter_item_embed_df = filter_item_embed_df.head(max_item_num)
    print("the number of filter_item_embed_df", len(filter_item_embed_df), "\n")
    print(filter_item_embed_df.head(5), "\n")

    filter_item_embed_df.drop(["index", "count"], axis=1, inplace=True)
    filter_item_embed_df = filter_item_embed_df.reset_index()
    filter_item_embed_df["index"] = filter_item_embed_df["index"].astype(int) + 2
    return user_df, filter_item_embed_df


def rebuild_item_index(user_df, filter_item_embed_df, output_path):
    # 获取新建索引和item的映射关系，打包存为csv文件
    item_id_df = filter_item_embed_df.drop("song_embedding", axis=1)
    item_id_file = os.path.join(output_path, "item_id.csv")
    item_id_df.to_csv(item_id_file, index=False, header=True, sep="|")
    print(f"the total number of item_id_df is: {len(item_id_df)}")
    print(item_id_df.head(5), "\n")

    # 获取新建索引和song embedding对应关系，打包存为pickle文件
    new_item2embed_df = filter_item_embed_df.drop("song_id", axis=1)
    new_item2embed_df.sort_values(by="index", ascending=True, ignore_index=True)
    print(f"the item2embed_df after sortting")
    print(new_item2embed_df.head(5), "\n")
    embeddings_series = new_item2embed_df["song_embedding"].apply(lambda x: x.split(","))
    embeddings_list = embeddings_series.apply(lambda x: [float(i) for i in x]).tolist()
    mean_embedding = np.mean(np.array(embeddings_list), axis=0).tolist()
    mean_embedding = [round(v, 6) for v in mean_embedding]
    pad_embedding = [0.0] * len(mean_embedding)
    new_embedding_list = [pad_embedding] + [mean_embedding] + embeddings_list
    print(f"lenght of new embedding list: {len(new_embedding_list)}")
    print(new_embedding_list[:5], "\n")

    new_item_embedding_file = os.path.join(output_path, "song_embeddings.pickle")
    with open(new_item_embedding_file, 'wb') as f:
        pickle.dump(torch.Tensor(new_embedding_list).float(), f)

    # 对用户行为数据中的item进行重新映射,并存为csv文件
    user_seq_df = user_df.assign(user_seq=user_df.positive_song_seq.str.split("#")).explode("user_seq")
    user_item2id = user_seq_df.merge(item_id_df, left_on="user_seq", right_on="song_id", how="left")
    user_item2id.fillna("1", inplace=True)
    user_item2id["index"] = user_item2id["index"].astype(int).astype(str)
    new_user_df = user_item2id.groupby("user_id")["index"].apply(lambda x: "|".join(x)).reset_index()
    batch_size = 1000000
    curr_line = 0
    total_line = len(new_user_df)
    print(f"the total number of user_df is: {total_line}")
    print(new_user_df.head(5), "\n")
    while curr_line < total_line:
        data_file = os.path.join(output_path, f"part_{curr_line}")
        print(data_file)
        new_user_df[curr_line:curr_line + batch_size].to_csv(data_file, index=False, header=False, sep="|")
        curr_line += batch_size
    return item_id_df["index"].tolist()


def get_negative_items(data_file, test_file, all_items, num_of_samples=100000, test_num=99):
    # 构建负样例
    user_items = defaultdict()

    lines = open(data_file).readlines()
    for line in lines:
        user, items = line.strip().split('|', 1)
        items = items.split('|')
        user_items[user] = set(items)

    curr = 0
    output = open(test_file, 'w', encoding='utf-8')
    for user, user_seq in user_items.items():
        if curr > num_of_samples:
            break
        curr += 1

        test_samples = set()
        while len(test_samples) < test_num:
            sample_ids = random.sample(all_items, test_num * 2)
            test_samples = (set(sample_ids) - user_seq) | test_samples
        test_samples_list = list(test_samples)
        test_samples_str = [str(v) for v in test_samples_list]
        output.write(user + '|' + '|'.join(test_samples_str[:test_num]) + '\n')

        if curr <= 5:
            print({user: '|'.join(test_samples_str[:test_num])})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user_file_path', required=True, type=str,
                        help='Please specify the path of user positive seq data')
    parser.add_argument('-i', '--item_file_path', required=True, type=str,
                        help='Please specify the path of item')
    parser.add_argument('-d', '--data_path', required=True, type=str,
                        help='Please specify the path of output')
    parser.add_argument('-m', '--max_item_num', required=False, type=int,
                        default=200000)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    start_time = time.time()
    user_df, filter_item_embed_df = filter_embedding_by_freq(args.user_file_path, args.item_file_path,
                                                             args.max_item_num)
    all_items = rebuild_item_index(user_df, filter_item_embed_df, args.data_path)

    print("all_items", all_items[:5])
    print("the number of total items: ", len(all_items))
    for file in os.listdir(args.data_path):
        if not file.startswith("part"):
            continue
        data_file = os.path.join(args.data_path, file)
        test_file = os.path.join(args.data_path, file + "_sample")
        print(data_file)
        get_negative_items(data_file, test_file, all_items, num_of_samples=100000, test_num=99)

    end_time = time.time()
    print(f"training time: {end_time - start_time}")


if __name__ == '__main__':
    main()
