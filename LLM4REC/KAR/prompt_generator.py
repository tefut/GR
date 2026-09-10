import argparse
import json
import os
import random
import re
from io import open
from typing import Dict, List, Tuple

import pandas as pd
from constant import FULL_ITEM_COLUMNS, GENDER_MAPPING, AGE_MAPPING
from constant_game import FULL_ITEM_GAME_COLUMNS


def load_csv_data(data_path: str, columns: List[str], sep: str) -> Tuple[pd.DataFrame, int]:
    remain_data = []
    error_line = 0
    with open(data_path, 'r') as f:
        for line in f.readlines():
            split_data = line.strip().split(sep)
            if len(split_data) != len(columns):
                error_line += 1
                print(split_data)
            remain_data.append(split_data)
    info = pd.DataFrame(data=remain_data, columns=columns)
    print('data shape', info.shape)
    print('error line', error_line)
    origin_num = info.shape[0] + error_line
    print(info.iloc[0])
    return info, origin_num


def load_orc_data(data_path: str) -> Tuple[pd.DataFrame, int]:
    info = pd.read_orc(data_path)
    print('load shape', info.shape)
    total_num = info.shape[0]
    info.replace('\\N', None, inplace=True)
    return info, total_num


def df2dict(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    line_keys = df.columns[1:]
    df_dict = {}
    for line in df.values:
        if not df_dict.__contains__(line[0]):
            df_dict[line[0]] = dict(zip(line_keys, line[1:]))
    return df_dict


def save_prompt(prompts, save_path, save_sep='|'):
    with open(save_path, 'w', encoding='utf-8') as file:
        print('prompt saved', save_path)
        for (index, prompt) in prompts:
            file.write(f'{index}{save_sep}{prompt}\r\n')


def format_prompt(info: Dict[str, Dict[str, str]], prompt_str: str = None) -> Dict[str, str]:
    pattern = re.compile(r'[{](.*?)[}]', re.S)
    key_col = re.findall(pattern, prompt_str)

    prompt = {}
    for info_key, info_value in info.items():
        col_value = {}
        for col in key_col:
            if col not in info_value:
                print(f'{col} not in {info_value.keys()}')
                break
            if col == "age":
                value = AGE_MAPPING.get(info_value.get(col, ''))
            elif col == "gender":
                value = GENDER_MAPPING.get(info_value.get(col, ''))
            else:
                value = info_value.get(col, '')
            if value is None or len(value) == 0:
                value = "未知"
            col_value[col] = value
        prompt[info_key] = prompt_str.format(**col_value)
    return prompt


def load_data_and_prompt(args):
    info_path = args.info_path
    all_columns = args.all_columns
    sep = args.sep
    save_path = args.prompt_save_path
    prompt_str = args.prompt_str
    num_lines = args.num_lines

    if args.orc_file.lower() == 'true':
        orc_file = True
    else:
        orc_file = False

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if "item" in info_path and len(all_columns) == 0:
        all_columns = FULL_ITEM_COLUMNS
        if "game" in info_path:
            all_columns = FULL_ITEM_GAME_COLUMNS

    data_paths = os.listdir(info_path)
    if '__SUCCESS' in data_paths:
        data_paths.remove('__SUCCESS')
    if '_SUCCESS' in data_paths:
        data_paths.remove('_SUCCESS')
    if '.ipynb_checkpoints' in data_paths:
        data_paths.remove('.ipynb_checkpoints')

    all_prompts = []
    total_num, valid_num, prompt_num, file_id = 0, 0, 0, 0
    for _, path in enumerate(data_paths):
        cur_path = os.path.join(info_path, path)
        print(f'load data from {cur_path}')
        if orc_file:
            info, num_per_file = load_orc_data(cur_path)
        else:
            if len(all_columns) == 0 or sep == '':
                print(f"the params all_columns and sep must be config in textfile!")
                return
            info, num_per_file = load_csv_data(cur_path, all_columns, sep)

        info = df2dict(info)
        total_num += num_per_file
        valid_num += len(info)
        prompts = format_prompt(info, prompt_str)

        all_prompts.extend(list(prompts.items()))
        while len(all_prompts) > num_lines:
            save_prompt(all_prompts[:num_lines], os.path.join(
                save_path, 'part_' + str(file_id)))
            all_prompts = all_prompts[num_lines:]
            prompt_num += num_lines
            file_id += 1
    if len(all_prompts) > 0:
        save_prompt(all_prompts, os.path.join(
            save_path, 'part_' + str(file_id)))
        prompt_num += len(all_prompts)

    print('total num', total_num, 'valid num', valid_num, 'prompt num', prompt_num)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--info_path', required=True,
                        type=str, help='Please specify the info path')
    parser.add_argument('-cu', '--all_columns', required=False, default='',
                        help='Please specify the user column names')
    parser.add_argument('-su', '--sep', required=False, type=str, default='|')
    parser.add_argument('-of', '--orc_file', required=False, type=str, default='true',
                        help='the file type of hive table')
    parser.add_argument('-ipt', '--prompt_str', required=True,
                        default='一个{user_age}的常驻{user_city}的{user_gender}用户，他喜欢的歌曲列表为{user_like_song_str}',
                        type=str,
                        help='prompt for data')
    parser.add_argument('-p', '--prompt_save_path', required=False, default='')
    parser.add_argument('-n', '--num_lines', required=False,
                        type=int, default=500000)
    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)

    train_config_file = "/opt/huawei/schedule-train/algorithm/train.config"
    with open(train_config_file, 'r', encoding='utf-8') as fin:
        train_config = json.load(fin)
    for param, config in train_config.items():
        if param in args.__dict__:
            args.__dict__[param] = config

    if args.all_columns == '':
        args.all_columns = []
    else:
        args.all_columns = args.all_columns.split(',')
    print('args', args)

    seed = 1234
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    load_data_and_prompt(args)


if __name__ == '__main__':
    main()
