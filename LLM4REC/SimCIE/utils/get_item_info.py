import argparse
import json
import os
import stat
import time

import pandas as pd


def write_to_file(save_file, mode='w'):
    flags = os.O_WRONLY | os.O_CREAT
    stats = stat.S_IWUSR | stat.S_IRUSR
    file_hander = os.fdopen(os.open(save_file, flags, stats), mode)
    return file_hander


def get_item_info(item_file_path, song_infos_file, fields_config):
    song_infos = []
    for fid, file in enumerate(os.listdir(item_file_path)):
        item_file = os.path.join(item_file_path, file)
        print("item_file", item_file)

        song_df = pd.read_orc(item_file)

        concat_columns = fields_config.get("info_columns", None)
        if concat_columns is not None:
            info_df = pd.concat([song_df[v] for v in concat_columns], axis=1)
        else:
            info_df = song_df

        if fid < 5:
            print(info_df.head(5))

        dropna_columns = fields_config.get("dropna_columns", None)
        if dropna_columns is not None:
            info_df.dropna(subset=dropna_columns, inplace=True)
        info_df.fillna("未知", inplace=True)
        song_infos.append(info_df)
        print("the length of song after preprocessing: ", len(info_df))

    song_infos_df = pd.concat(song_infos, axis=0)
    print(song_infos_df.head(5))
    print(song_infos_df.columns)
    print("the total length of item: ", len(song_infos_df))

    song_infos_output = write_to_file(song_infos_file, 'w')
    song_infos_df.to_csv(song_infos_output, index=False, header=False)
    return song_infos_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--item_file_path', required=True, type=str)
    parser.add_argument('-o', '--item_info_file', required=True, type=str)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    mtp_train_config_file = "/opt/huawei/schedule-train/algorithm/train.config"
    if os.path.exists(mtp_train_config_file):
        with open(mtp_train_config_file, 'r', encoding='utf-8') as fin:
            train_config = json.load(fin)
        fields_config = train_config
    else:
        fields_config = {
            "info_columns": ["song_id", "song_name", "album_name", "artist_name_set", "song_composer"],
            "dropna_columns": ["song_id", "song_name"]
        }
    print(f"fields_config: {fields_config}")

    start_time = time.time()

    data_path = os.path.dirname(args.item_info_file)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        print("make dir: ", data_path)

    get_item_info(args.item_file_path, args.item_info_file, fields_config)

    end_time = time.time()
    print(f"time: {end_time - start_time}")


if __name__ == '__main__':
    main()
