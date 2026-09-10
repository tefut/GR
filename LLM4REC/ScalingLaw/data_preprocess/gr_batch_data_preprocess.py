import argparse
import json
import os
import time
from collections import defaultdict
import multiprocessing
from functools import partial

import tensorflow as tf
import pandas as pd
import numpy as np

from climber_batch_data_preprocess import create_tf_feature_description
from climber_batch_data_preprocess import get_batch_song_info


def batch_parse_data(numpy_batch, dict_data):
    # change to pass in args
    # add user_id in the list, 得到和林曦中洲形式完全一样的数据
    user_info_key = ["age", "gender", "province", "city", "device_price", 
                     "device_series_name", "city_level", "product_name"]

    user_info = np.hstack(tuple(numpy_batch[key] for key in user_info_key))
    user_info = [",".join(v.astype(str)) for v in user_info]

    sequence_rating = [",".join(v[v != 0].astype(str)) for v in numpy_batch["sequence_ratings"]]
    sequence_column = [",".join(v[v != 0].astype(str)) for v in numpy_batch["sequence_column"]]

    dict_data["user_id"].extend(user_info)
    dict_data["sequence_ratings"].extend(sequence_rating)
    dict_data["sequence_column"].extend(sequence_column)

    # change to pass in args
    sequence_info_key = ["sequence_artist_seq", "sequence_langua_seq",
                         "sequence_album_seq", "sequence_genre_seq", "sequence_scene_seq", "sequence_mood_seq",
                         "sequence_theme_seq"]

    
    item_id_seq, sequence_timestamps, sequence_song_info = get_batch_song_info("sequence_item_ids",
                                                                                          sequence_info_key,
                                                                                          "sequence_timestamps",
                                                                                          numpy_batch)

    kv = {
          "sequence_item_ids": item_id_seq, "sequence_info": sequence_song_info,
          "sequence_timestamps": sequence_timestamps, }

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

    dict_data = defaultdict(list)

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
