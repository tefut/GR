import argparse
import os
import stat

import pandas as pd


def write_to_file(save_file, mode='w'):
    flags = os.O_WRONLY | os.O_CREAT
    stats = stat.S_IWUSR | stat.S_IRUSR
    file_hander = os.fdopen(os.open(save_file, flags, stats), mode)
    return file_hander


def get_data_from_dir(data_path, sep, target_column, min_item_num, output_file):
    output = write_to_file(output_file, "w")
    total_number = 0
    for file in os.listdir(data_path):
        data_file = os.path.join(data_path, file)
        print(data_file)
        if file.endswith("csv"):
            df = pd.read_csv(data_file)
        else:
            df = pd.read_orc(data_file)
        if target_column not in df.columns:
            print(f"{target_column} not in data.columns: {df.columns}")
            return
        df.dropna(subset=[target_column], inplace=True)
        total_number += len(df)
        for line in df[target_column].tolist():
            line = line.strip().split(sep)
            if len(line) < min_item_num:
                continue
            output.write(" ".join(line) + "\n")
    print(f"the length of data: {total_number}")


def get_data_from_file(data_path, sep, target_column, min_item_num, output_file):
    output = write_to_file(output_file, "w")
    if data_path.endswith("csv"):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_orc(data_path)

    if target_column not in df.columns:
        print(f"target_column not in data.columns: {df.columns}")
        return

    df.dropna(subset=[target_column], inplace=True)
    for line in df[target_column].tolist():
        line = line.strip().split(sep)
        if len(line) < min_item_num:
            continue
        output.write(" ".join(line) + "\n")
    print(f"the length of data: {len(df)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', required=True, type=str)
    parser.add_argument('-o', '--output_file', required=True, type=str)
    parser.add_argument('-s', '--sep', required=True, type=str)
    parser.add_argument('-c', '--target_column', required=True, type=str)
    parser.add_argument('-m', '--min_item_num', required=False, type=int, default=1)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    if os.path.isdir(args.data_path):
        get_data_from_dir(args.data_path, args.sep, args.target_column, args.min_item_num, args.output_file)
    else:
        get_data_from_file(args.data_path, args.sep, args.target_column, args.min_item_num, args.output_file)


if __name__ == '__main__':
    main()
