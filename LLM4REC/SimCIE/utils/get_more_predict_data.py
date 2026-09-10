import argparse
import json
import os
import pickle
import stat
import time

import pandas as pd
import torch


def get_prompt(row, fields_length, prompt_str):
    for field, length in fields_length.items():
        value = str(row[field])[:int(length)]
        prompt_str = prompt_str.replace(field, value)
    return prompt_str


def get_more_predict_item(play_to_positive_file, item_info_file, index_to_item_file, fields_config, output_file):
    # 加载item文本信息；
    info_columns = fields_config.get("info_columns", None)
    item_info_df = pd.read_csv(item_info_file, names=info_columns)
    item_info_df.rename(columns={item_info_df.columns[0]: "item_id"}, inplace=True)
    item_info_df["item_id"] = item_info_df["item_id"].astype(str)
    item_info_df.fillna("未知", inplace=True)
    print_df_info(item_info_df, "item文本信息")

    # 加载play_to_positive_item文件
    play_to_positive_df = pd.read_csv(play_to_positive_file, names=["item_id", "positive_item", "score"], sep="|")
    play_to_positive_df["item_id"] = play_to_positive_df["item_id"].astype(str)
    play_to_positive_df["positive_item"] = play_to_positive_df["positive_item"].astype(str)
    print_df_info(play_to_positive_df, "完播item 召回 强正反馈item")

    # 获取完播item的文本信息
    combine_info_df = pd.merge(play_to_positive_df, item_info_df, on="item_id", how='left')
    dropna_columns = list(set(combine_info_df.columns) & set(fields_config.get("dropna_columns", [])))
    if len(dropna_columns) != 0:
        combine_info_df.dropna(subset=dropna_columns, inplace=True)
    print_df_info(combine_info_df, "完播item的文本信息")

    # 构造prompt
    fields_length = fields_config.get("fields_length", {})
    prompt_str = fields_length.pop("prompt", None)
    combine_info_df["info"] = combine_info_df.apply(get_prompt, args=(fields_length, prompt_str,), axis=1)
    print_df_info(combine_info_df, "完播item的prompt")

    # 获取强正反馈item对应的索引
    index_to_item_df = pd.read_csv(index_to_item_file, names=["index", "positive_item"], sep="|")
    index_to_item_df["positive_item"] = index_to_item_df["positive_item"].astype(str)
    index_prompt_df = pd.merge(combine_info_df, index_to_item_df, on="positive_item", how="left")
    index_prompt_df.dropna(subset=["index"], inplace=True)
    print_df_info(index_prompt_df, "强正反馈item对应的索引")

    # 构造预测数据
    data_df = pd.concat(
        [index_prompt_df['index'], index_prompt_df['item_id'], index_prompt_df['info'],
         index_prompt_df['positive_item']], axis=1)
    print_df_info(data_df, "预测数据")
    data_df.to_csv(output_file, index=False)


def print_df_info(df, df_name):
    print(f"----------------------- {df_name} -----------------------------------")
    print(f"the length of df: {len(df)}")
    print(f"the columns of df:{df.columns}")
    print("sample of df: ")
    print(df.head(5))
    print(f"---------------------------------------------------------------------------------")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--play_to_positive_file', required=True, type=str)
    parser.add_argument('-i', '--item_info_file', required=True, type=str)
    parser.add_argument('-m', '--index_to_item_file', required=True, type=str)
    parser.add_argument('-o', '--output_file', required=True, type=str)

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
            "info_columns": ["item_id", "song_name", "album_name", "artist_name", "song_composer", "label", "language",
                             "flag", "release_time"],
            "dropna_columns": ["item_id", "song_name"],
            "fields_length": {
                "song_name": 20,
                "album_name": 10,
                "artist_name": 10,
                "song_composer": 10,
                "language": 10,
                "label": 10,
                "prompt": "歌名:《song_name》，歌手：artist_name，专辑名：《album_name》，作曲者：song_composer，语种：language，标签：label，歌曲id："
            }
        }

    print(f"fields_config: {fields_config}")

    get_more_predict_item(args.play_to_positive_file, args.item_info_file, args.index_to_item_file, fields_config,
                          args.output_file)

    end_time = time.time()
    print(f"time: {end_time - start_time}")


if __name__ == '__main__':
    main()
