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


def get_feature_config(model_config_file):
    features = {}
    with open(model_config_file, encoding='utf-8') as fin:
        data = json.load(fin)
        for feature in data.get("features"):
            feat_name = feature.get("name")
            features[feat_name] = feature
    return features


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


def append_to_orc(dict_data, file_path, index):
    file_path = f"{file_path}.{index}.orc"
    df = pd.DataFrame(dict_data)
    df.to_orc(file_path)


def get_DLRM_data(data_file, output_path, feature_description, batch_size, chunk_size):
    tf_data_file_list = tf.data.TFRecordDataset.from_tensor_slices([data_file])
    train_dataset = tf.compat.v1.data.TFRecordDataset(tf_data_file_list, compression_type="GZIP")

    file_name = os.path.basename(data_file)
    output_file = os.path.join(output_path, file_name.replace(".tfrecord.gz", ""))

    def _parse_example(example_string):
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        return feature_dict

    train_dataset = train_dataset.map(_parse_example)
    dataset = train_dataset.batch(batch_size=batch_size)
    dict_data = {}
    for key in feature_description:
        dict_data[key] = []

    for i, batch in enumerate(dataset):
        if i % chunk_size == 0 and i != 0:
            append_to_orc(dict_data, output_file, str(i))
            del dict_data
            dict_data = {}
            for key in feature_description:
                dict_data[key] = []
        for key, value in batch.items():
            dict_data[key].extend(value.numpy().tolist())

    append_to_orc(dict_data, output_file, "end")
    print(f"parse file: {data_file} success!")
    del dict_data


def get_train_data(data_file, output_path, feature_description, batch_size, chunk_size):
    # 导入训练数据
    tf_data_file_list = tf.data.TFRecordDataset.from_tensor_slices([data_file])
    train_dataset = tf.compat.v1.data.TFRecordDataset(tf_data_file_list, compression_type="GZIP")

    # 对tfrecord数据进行解析
    def _parse_example(example_string):
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        return feature_dict

    file_name = os.path.basename(data_file)
    output_file = os.path.join(output_path, file_name.replace(".tfrecord.gz", ""))

    train_dataset = train_dataset.map(_parse_example)
    dataset = train_dataset.batch(batch_size=batch_size)
    dict_data = {}
    for key in feature_description:
        dict_data[key] = []

    for i, batch in enumerate(dataset):
        if i % chunk_size == 0 and i != 0:
            append_to_orc(dict_data, output_file, str(i))
            del dict_data
            dict_data = {}
            for key in feature_description:
                dict_data[key] = []
        for key, value in batch.items():
            if key in ["genre", "scene", "theme", "mood", "langua", "eras", "artist"]:
                dict_data[key].extend([[int(v[0])] for v in value.numpy().tolist()])
            else:
                dict_data[key].extend(value.numpy().tolist())

    append_to_orc(dict_data, output_file, "end")
    del dict_data
    print(f"parse file: {data_file} success!")


def get_valid_data(data_file, feature_description, features, batch_size):
    # 导入训练数据
    print(data_file)
    tf_data_file_list = tf.data.TFRecordDataset.from_tensor_slices([data_file])
    train_dataset = tf.compat.v1.data.TFRecordDataset(tf_data_file_list, compression_type="GZIP")

    # 对tfrecord数据进行解析
    def _parse_example(example_string):
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        return feature_dict

    train_dataset = train_dataset.map(_parse_example)
    dataset = train_dataset.batch(batch_size=batch_size)

    dict_data = {}
    for key in feature_description:
        dict_data[key] = []

    for batch in dataset:
        for key, value in batch.items():
            length = features[key].get('length')
            if length == 1:
                dict_data[key].extend(np.squeeze(value.numpy(), axis=1))
            elif key in ["genre", "scene", "theme", "mood", "langua", "eras", "artist"]:
                dict_data[key].extend([int(v[0]) for v in value.numpy().tolist()])
            else:
                dict_data[key].extend([",".join(map(str, v)) for v in value.numpy().tolist()])

    return pd.DataFrame(dict_data)


def agg_apply(group, features):
    result = {}
    for col in group:
        region = features[col].get("region").split(",")
        if "user" in region:
            result[col] = group[col].iloc[0]
        else:
            result[col] = group[col].tolist()
    return pd.Series(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True, type=str)
    parser.add_argument('-m', '--model_config_file', required=True, type=str)
    parser.add_argument('-o', '--output_path', required=True, type=str)
    parser.add_argument('-b', '--batch_size', required=True, type=int)
    parser.add_argument('-n', '--num_processes', required=True, type=int)
    parser.add_argument('-c', '--chunk_size', required=True, type=int)
    parser.add_argument('-md', '--mode', required=True, type=str)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    start_time = time.time()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
        print("make dir: ", args.output_path)

    # 构造feature_description
    feature_description = create_tf_feature_description(args.model_config_file)
    features = get_feature_config(args.model_config_file)
    file_paths = [os.path.join(args.input_path, v) for v in os.listdir(args.input_path) if v.startswith("part")]

    if args.mode == "train":
        chunk_size = args.chunk_size // args.batch_size
        partial_process_file = partial(get_train_data, output_path=args.output_path,
                                       feature_description=feature_description,
                                       batch_size=args.batch_size, chunk_size=chunk_size)
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            results = pool.map(partial_process_file, file_paths)
            for result in results:
                if result:
                    print(f"文件处理完成，结果保存到 {result}")
    elif args.mode == "DLRM":
        chunk_size = args.chunk_size // args.batch_size
        partial_process_file = partial(get_DLRM_data, output_path=args.output_path,
                                       feature_description=feature_description,
                                       batch_size=args.batch_size, chunk_size=chunk_size)
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            results = pool.map(partial_process_file, file_paths)
            for result in results:
                if result:
                    print(f"文件处理完成，结果保存到 {result}")
    elif args.mode == "valid":
        # 创建偏函数
        partial_process_file = partial(get_valid_data, feature_description=feature_description, features=features,
                                       batch_size=args.batch_size)
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            results = pool.map(partial_process_file, file_paths)
            print("the length of files: ", len(results))

        data_df = pd.concat(results, ignore_index=True)
        print(data_df.head(5))

        total_df = data_df.groupby("user_id").apply(agg_apply, features)
        print(total_df.head(5))
        print(len(total_df))

        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
            print("make dir: ", args.output_path)

        for i in range(0, len(total_df), args.chunk_size):
            end = min(i + args.chunk_size, len(total_df))
            chunk = total_df.iloc[i:end]

            output_file = os.path.join(args.output_path, f"output_path_{end}.csv")
            chunk.to_orc(output_file, index=False)
    else:
        print("error! mode must be train or valid!")
    end_time = time.time()
    print(f"time: {end_time - start_time}")


if __name__ == '__main__':
    main()
