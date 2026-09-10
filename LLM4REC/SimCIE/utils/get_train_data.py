import argparse
import json
import os
import pickle
import stat
import time

import pandas as pd
import torch


def write_to_file(save_file, mode='w'):
    flags = os.O_WRONLY | os.O_CREAT
    stats = stat.S_IWUSR | stat.S_IRUSR
    file_hander = os.fdopen(os.open(save_file, flags, stats), mode)
    return file_hander


def combine_file(data_path, output_file):
    total_df = []
    for file in os.listdir(data_path):
        data_file = os.path.join(data_path, file)
        print(data_file)
        if file.endswith("csv"):
            df = pd.read_csv(data_file)
        else:
            df = pd.read_orc(data_file)
        total_df.append(df)
    data_df = pd.concat(total_df)
    data_df.to_csv(output_file, index=False)


def get_prompt(row, fields_length, prompt_str):
    for field, length in fields_length.items():
        value = str(row[field])[:int(length)]
        prompt_str = prompt_str.replace(field, value)
    return prompt_str


def get_item_info(item_embeddings_file, item_info_file, fields_config):
    # 加载item文本信息；
    info_columns = fields_config.get("info_columns", None)
    item_info_df = pd.read_csv(item_info_file, names=info_columns)
    item_info_df.rename(columns={item_info_df.columns[0]: "item_id"}, inplace=True)
    item_info_df["item_id"] = item_info_df["item_id"].astype(str)
    item_info_df.fillna("未知", inplace=True)
    print("加载item文本信息")
    print("length of item_info_df", len(item_info_df))
    print(item_info_df.columns)
    print("the dataframe of item info")
    print(item_info_df.head(5))

    # 加载item预训练embedding信息
    item_embeddings_df = pd.read_csv(item_embeddings_file, names=["item_id", "item_embedding", "cluster_id"])
    item_embeddings_df["item_id"] = item_embeddings_df["item_id"].astype(str)
    print("加载item预训练embedding信息")
    print("length of item_embeddings_df", len(item_embeddings_df))
    print("the dataframe of item embeddings")
    print(item_embeddings_df.head(5))

    # 合并item文本和embedding
    combine_info_df = pd.merge(item_embeddings_df, item_info_df, on="item_id", how='left')
    dropna_columns = list(set(combine_info_df.columns) & set(fields_config.get("dropna_columns", [])))
    if len(dropna_columns) != 0:
        combine_info_df.dropna(subset=dropna_columns, inplace=True)
    print("合并item文本和embedding")
    print("length of item embedding combine with item info", len(combine_info_df))
    print("the dataframe of item embedding combine with item info")
    print(combine_info_df.tail(5))

    # 构造prompt
    fields_length = fields_config.get("fields_length", {})
    prompt_str = fields_length.pop("prompt", None)
    print("prompt_str: ", prompt_str)
    combine_info_df["info"] = combine_info_df.apply(get_prompt, args=(fields_length, prompt_str,), axis=1)
    print("构造prompt")
    print(combine_info_df.tail(5))

    # 数据清洗，清除文本信息为空的item
    prompt_df = combine_info_df.dropna(subset=["item_id", "info"]).reset_index(drop=True)
    prompt_df = prompt_df.reset_index()
    print("数据清洗，清除文本信息为空的item")
    print("length of item_info after process: ", len(prompt_df))
    print("the dataframe of item_info")
    print(prompt_df.tail(5))
    return prompt_df


def get_predict_item(item_info_df, index_to_item_file, item_embedding_file, predict_file):
    # 构建index_to_itemid文件
    index_to_item_df = pd.concat([item_info_df['index'], item_info_df['item_id']], axis=1)
    print("构建index_to_itemid文件")
    print("length of index_to_item_df", len(index_to_item_df))
    print("the dataframe of index_to_item_df")
    print(index_to_item_df.head(5))
    index_to_item_df.to_csv(index_to_item_file, index=False, header=True, sep="|")

    # 构建item embedding pickle文件
    item_embedding_series = item_info_df.pop("item_embedding").apply(lambda x: x.split(","))
    item_embedding_list = item_embedding_series.apply(lambda x: [float(i) for i in x]).tolist()
    item_embedding_output = write_to_file(item_embedding_file, 'wb')
    pickle.dump(torch.Tensor(item_embedding_list).float(), item_embedding_output)

    # 构造预测数据，用于导出item embedding matrix
    data_df = pd.concat(
        [item_info_df['index'], item_info_df['item_id'], item_info_df['info'], item_info_df['cluster_id']], axis=1)
    print("构造预测数据，用于导出item embedding matrix")
    print("length of total item", len(data_df))
    print("the dataframe of total item")
    print(data_df.head(5))

    data_df.to_csv(predict_file, index=False)
    return data_df


def get_train_data_with_kmean(df, train_file, number):
    # 构造训练样例，从每个分组中随机抽取两个句子，形成正样例
    results = []
    data = df.groupby('cluster_id')
    for _, group in data:
        source_items_df = group.sample(n=number, replace=True).reset_index(drop=True)
        target_items_df = group.sample(n=number, replace=True).reset_index(drop=True)
        source_columns_name = [name + "_x" for name in source_items_df.columns]
        target_columns_name = [name + "_y" for name in target_items_df.columns]

        source_items_df.columns = source_columns_name
        target_items_df.columns = target_columns_name
        result = pd.concat([source_items_df, target_items_df], axis=1)
        results.append(result)

    train_df = pd.concat(results, axis=0)
    print(train_df.head(5))
    train_df.to_csv(train_file, index=False)


def get_train_data_with_sim(df, sim_score_file, train_file, eval_file, top_k, top_p):
    # 根据item之间的相似分数进行抽样

    # 获取item之间的相似分数，并过滤分数小于top_p的item对
    sort_df = pd.read_csv(sim_score_file, sep=",", names=["item_id", "target_id", "score"])
    sort_df = sort_df[sort_df["score"] > top_p]  # 根据相似分数筛选
    print(f"根据相似分数大于{top_p}筛选样例")
    print("length of item after filtring by score", len(sort_df))
    print("the dataframe of score item")
    print(sort_df.head(5))
    sort_df['target_id'] = sort_df['target_id'].astype(str)
    sort_df['item_id'] = sort_df['item_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)

    # 获取目标item的文本信息
    source_df = sort_df.merge(df, how="left", on="item_id")
    # 获取正样例的文本信息
    target_df = source_df.merge(df, how="left", left_on="target_id", right_on="item_id")
    target_df.drop(["target_id", "score"], axis=1, inplace=True)
    # 删除文本信息为空的样例
    target_df.dropna(subset=["info_x", "info_y"], inplace=True)
    print("数据清洗")
    print("the length of sample after removing none", len(target_df))
    print(target_df.head(5))

    # 对目标item分组，从每个分组中抽取top_k个样例来构造训练数据
    train_df = pd.concat([group.head(top_k) for _, group in target_df.groupby("item_id_x")])  # 取每个item的正样例个数
    train_df = train_df.reindex(
        columns=["index_x", "item_id_x", "info_x", "cluster_id_x", "index_y", "item_id_y", "info_y", "cluster_id_y"])
    print(f"对目标item分组，从每个分组中抽取{top_k}个样例来构造训练数据")
    print("the length of train data", len(train_df))
    print("the sample of train data")
    print(train_df.head(15))
    eval_df = train_df[int(0.8 * len(train_df)):]
    train_df = train_df[:int(0.8 * len(train_df))]
    train_df.to_csv(train_file, index=False)
    eval_df.to_csv(eval_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', required=True, type=str)
    parser.add_argument('-i', '--item_info_file', required=True, type=str)
    parser.add_argument('-m', '--mode', required=False, type=str, default="kmean")
    parser.add_argument('-n', '--number', required=False, type=int, default=500)
    parser.add_argument('-s', '--sim_score_file', required=False, type=str)
    parser.add_argument('-p', '--top_p', required=False, type=float, default=0.7)
    parser.add_argument('-k', '--top_k', required=False, type=int, default=5)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    start_time = time.time()

    mtp_train_config_file = "/opt/huawei/schedule-train/algorithm/train.config"
    if os.path.exists(mtp_train_config_file):
        with open(mtp_train_config_file, 'r', encoding='utf-8') as fin:
            train_config = json.load(fin)
        fields_config = train_config
    else:
        fields_config = {
            "info_columns": ["song_id", "album_name", "artist_name", "composer", "song_name"],
            "dropna_columns": ["song_id", "song_name"],
            "fields_length": {
                "song_name": 20,
                "album_name": 10,
                "artist_name_set": 10,
                "song_composer": 10,
                "prompt": "歌名:《song_name》，歌手：artist_name，专辑名：《album_name》，作曲者：composer，歌曲id："
            }
        }

    print(f"fields_config: {fields_config}")

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
        print("make dir: ", args.data_path)

    data_file = os.path.join(args.data_path, "kmean_embedding.csv")
    item_info_file = args.item_info_file
    index_to_item_file = os.path.join(args.data_path, "index_to_item.csv")
    item_embedding_file = os.path.join(args.data_path, "item_embedding.pickle")
    train_file = os.path.join(args.data_path, "train.csv")
    eval_file = os.path.join(args.data_path, "eval.csv")
    predict_file = os.path.join(args.data_path, "predict.csv")

    if os.path.isdir(item_info_file):
        new_item_info_file = os.path.join(args.data_path, "combine_item_info.csv")
        combine_file(item_info_file, new_item_info_file)
        print(f"there are multiple files of item info, merge them into {new_item_info_file}")
        item_info_file = new_item_info_file

    item_info_df = get_item_info(data_file, item_info_file, fields_config)
    df = get_predict_item(item_info_df, index_to_item_file, item_embedding_file, predict_file)

    if args.mode == "kmean":
        get_train_data_with_kmean(df, train_file, args.number)
    else:
        get_train_data_with_sim(df, args.sim_score_file, train_file, eval_file, args.top_k, args.top_p)

    end_time = time.time()
    print(f"time: {end_time - start_time}")


if __name__ == '__main__':
    main()
