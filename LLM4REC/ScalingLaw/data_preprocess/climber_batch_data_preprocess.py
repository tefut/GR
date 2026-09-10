import argparse
import json
import os
import time
from collections import OrderedDict
import multiprocessing
from functools import partial

import tensorflow as tf
import pandas as pd
import numpy as np


def create_tf_feature_description(feature_config_file):
    with open(feature_config_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    feature_config_data = data["features"]

    TYPE_TO_TF_TYPE = {
        "int": "int64",
        "long": "int64",
        "float": "float32",
        "double": "float32"
    }

    # 构造tf feature样例
    tf_feature_dict = OrderedDict()
    for feature in feature_config_data:
        shape = feature.get("length")
        dtype = TYPE_TO_TF_TYPE.get(feature.get("value_type"), "string")
        tf_feature_dict[feature.get("name")] = tf.io.FixedLenFeature(shape=shape, dtype=dtype)
    return tf_feature_dict


def get_batch_song_info(song_seq_key, song_info_keys, timestamp_key, numpy_batch):
    song_seq = [v[v != 0] for v in numpy_batch[song_seq_key]]
    timestamp = [v[v != 0] for v in numpy_batch[timestamp_key].astype(int)]

    min_len = [min(a.shape[0], b.shape[0]) for a, b in zip(song_seq, timestamp)]

    song_seq = [v[:length] for v, length in zip(song_seq, min_len)]
    timestamp = [v[:length] for v, length in zip(timestamp, min_len)]

    for key in song_info_keys:
        numpy_batch[key] = np.array([numpy_batch[key][i, : min_len[i]] for i in range(len(min_len))])

    a = np.stack(tuple(numpy_batch[key] for key in song_info_keys), axis=-1)
    song_info = [np.stack(v, axis=-1).flatten() for v in a]

    return song_seq, timestamp, song_info


def batch_parse_data(numpy_batch, dict_data):
    user_info_key = ["age", "gender", "province", "city", "device_price", "device_series_name"]
    user_info = np.hstack(tuple(numpy_batch[key] for key in user_info_key))
    user_info = [",".join(v.astype(str)) for v in user_info]

    sequence_rating = [",".join(v[v != 0].astype(str)) for v in numpy_batch["sequence_ratings"]]
    sequence_column = [",".join(v[v != 0].astype(str)) for v in numpy_batch["sequence_column"]]

    dict_data["user_id"].extend(user_info)
    dict_data["sequence_ratings"].extend(sequence_rating)
    dict_data["sequence_column"].extend(sequence_column)

    play_song_info_key = ["play_song_artist_seq", "play_song_langua_seq", "play_week_sequence", "play_hour_sequence",
                          "play_song_album_seq", "play_song_genre_seq", "play_song_scene_seq", "play_song_mood_seq",
                          "play_song_theme_seq"]
    profile_song_info_key = ["profile_song_artist_seq", "profile_song_langua_seq", "profile_week_sequence",
                             "profile_hour_sequence", "profile_song_album_seq", "profile_song_genre_seq",
                             "profile_song_scene_seq", "profile_song_mood_seq", "profile_song_theme_seq"]
    search_song_info_key = ["search_song_artist_seq", "search_song_langua_seq", "search_week_sequence",
                            "search_hour_sequence", "search_song_album_seq", "search_song_genre_seq",
                            "search_song_scene_seq", "search_song_mood_seq", "search_song_theme_seq"]
    sequence_info_key = ["sequence_artist_seq", "sequence_langua_seq", "sequence_week_seq", "sequence_hour_seq",
                         "sequence_album_seq", "sequence_genre_seq", "sequence_scene_seq", "sequence_mood_seq",
                         "sequence_theme_seq"]

    play_song_seq, play_song_timestamps, play_song_info = get_batch_song_info("play_song_seq", play_song_info_key,
                                                                              "play_song_timestamps", numpy_batch)
    profile_song_seq, profile_song_timestamps, profile_song_info = get_batch_song_info("profile_song_seq",
                                                                                       profile_song_info_key,
                                                                                       "profile_song_timestamps",
                                                                                       numpy_batch)
    search_song_seq, search_song_timestamps, search_song_info = get_batch_song_info("search_song_seq",
                                                                                    search_song_info_key,
                                                                                    "search_song_timestamps",
                                                                                    numpy_batch)
    sequence_song_seq, sequence_song_timestamps, sequence_song_info = get_batch_song_info("sequence_item_ids",
                                                                                          sequence_info_key,
                                                                                          "sequence_timestamps",
                                                                                          numpy_batch)

    kv = {"play_item_ids": play_song_seq, "play_info": play_song_info, "play_timestamps": play_song_timestamps,
          "profile_item_ids": profile_song_seq, "profile_info": profile_song_info,
          "profile_timestamps": profile_song_timestamps,
          "search_item_ids": search_song_seq, "search_info": search_song_info,
          "search_timestamps": search_song_timestamps,
          "sequence_item_ids": sequence_song_seq, "sequence_info": sequence_song_info,
          "sequence_timestamps": sequence_song_timestamps, }

    for k, v in kv.items():
        new_v = [",".join(d.astype(str)) for d in v]
        dict_data[k].extend(new_v)


def read_data(data_file, output_path, feature_description, batch_size):
    # 导入训练数据
    tf_data_file_list = tf.data.TFRecordDataset.from_tensor_slices([data_file])
    train_dataset = tf.compat.v1.data.TFRecordDataset(tf_data_file_list, compression_type="GZIP")

    # 对tfrecord数据进行解析
    def _parse_example(example_string):
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        return feature_dict

    dict_data = {}
    for key in ["user_id", "sequence_ratings", "sequence_column", "sequence_item_ids", "sequence_info",
                "sequence_timestamps",
                "play_item_ids", "play_info", "play_timestamps",
                "profile_item_ids", "profile_info", "profile_timestamps",
                "search_item_ids", "search_info", "search_timestamps"]:
        if key not in dict_data:
            dict_data[key] = []

    train_dataset = train_dataset.map(_parse_example)
    dataset = train_dataset.batch(batch_size)
    for batch in dataset:
        numpy_batch = {key: value.numpy() for key, value in batch.items()}

        batch_parse_data(numpy_batch, dict_data)

    file_name = os.path.basename(data_file)
    output_file = os.path.join(output_path, file_name.replace(".tfrecord.gz", ".orc"))
    df = pd.DataFrame(dict_data)
    df.to_orc(output_file)
    print(f"parse file: {data_file} success!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True, type=str)
    parser.add_argument('-m', '--model_config_file', required=True, type=str)
    parser.add_argument('-o', '--output_path', required=True, type=str)
    parser.add_argument('-b', '--batch_size', required=True, type=int)
    parser.add_argument('-n', '--num_processes', required=True, type=int)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    start_time = time.time()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
        print("make dir: ", args.output_path)

    # 构造feature_description
    feature_description = create_tf_feature_description(args.model_config_file)
    file_paths = [os.path.join(args.input_path, v) for v in os.listdir(args.input_path) if v.startswith("part")]

    # 创建偏函数
    partial_process_file = partial(read_data, output_path=args.output_path, feature_description=feature_description,
                                   batch_size=args.batch_size)
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        results = pool.map(partial_process_file, file_paths)
        for result in results:
            if result:
                print(f"文件处理完成，结果保存到 {result}")

    end_time = time.time()
    print(f"time: {end_time - start_time}")


if __name__ == '__main__':
    main()
