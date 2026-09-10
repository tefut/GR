import argparse
import json
import os
from collections import OrderedDict

import tensorflow as tf


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


def read_data(data_files, model_config_file, number_of_samples):
    # 构造feature_description
    feature_description = create_tf_feature_description(model_config_file)
    print("\nfeature_description:")
    print(feature_description, '\n')

    # 导入训练数据
    tf_data_file_list = tf.data.TFRecordDataset.from_tensor_slices(data_files)
    train_dataset = tf.compat.v1.data.TFRecordDataset(tf_data_file_list, compression_type="GZIP")

    # 对tfrecord数据进行解析
    def _parse_example(example_string):
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        return feature_dict

    train_dataset = train_dataset.map(_parse_example)
    iterator = train_dataset.make_one_shot_iterator()

    for _ in range(number_of_samples):
        element = iterator.get_next()
        for key, value in element.items():
            print(key, value, '\n')

    print("complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number_of_samples', required=True, type=int, default=10,
                        help='import argparse')
    parser.add_argument('-p', '--data_path', required=True, help='the path of dataset')
    parser.add_argument('-m', '--model_config_file', required=True, type=str,
                        help='Please specify the path of model_config.json')
    parser.add_argument('-f', '--file_name', required=False, type=str, default=None,
                        help='Please specify the data file')
    args, unknown = parser.parse_known_args()

    print('unknown arguments: ', unknown)
    train_config_file = "/opt/huawei/schedule-train/algorithm/train.config"
    with open(train_config_file, 'r', encoding='utf-8') as fin:
        train_config = json.load(fin)
    for param, config in train_config.items():
        if param in args.__dict__:
            args.__dict__[param] = config
    print('arguments: ', args)

    if os.path.isdir(args.data_path):
        print(f"the data path is {args.data_path}")
    else:
        print(f"{args.data_path} is not a directory! ")
        exit()

    if args.file_name is None:
        for file in os.listdir(args.data_path):
            if file.endswith("tfrecord.gz"):
                args.file_name = file
                break
    if args.file_name is None:
        print("there are not file in the data path! exit.")
        exit()
    data_files = [os.path.join(args.data_path, args.file_name)]
    print(f"the samples is from {data_files[0]}")

    read_data(data_files, args.model_config_file, args.number_of_samples)


if __name__ == '__main__':
    main()
