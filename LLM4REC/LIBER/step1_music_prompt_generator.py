import argparse
import os
import random
import time
from io import open
from typing import Dict, List, Tuple, Set

import pandas as pd
from constant import FULL_NECESSARY_USER_COLUMNS, FULL_NECESSARY_ITEM_COLUMNS, SONG_NAME, SONG_ARTIST_NAME_SET, \
    SONG_ARTIST_SEP, ALBUM_NAME, SONG_GENRES_TAG_NAME, SONG_LANGUA_TAG_NAME, FULL_ITEM_COLUMN_SEP, FULL_ITEM_COLUMNS, \
    AGE_MAPPING, GENDER_MAPPING, USER_SONG_SET_SEP


def load_item_data(data_path: str, columns: List[str], sep: str,
                   selected_columns: List[str] = None) -> pd.DataFrame:
    remain_data = []
    error_line = 0
    with open(data_path, 'r') as f:
        for line in f.readlines():
            split_data = line.strip().split(sep)
            if len(split_data) != len(columns):
                error_line += 1
            remain_data.append(split_data)
    info = pd.DataFrame(data=remain_data, columns=columns)
    print('shape', info.shape)
    print('error line', error_line)
    origin_num = info.shape[0] + error_line
    print(info.iloc[0])
    if selected_columns:
        info = info[selected_columns]
        print('necessary shape', info.shape)
        print(info.iloc[0])
    info = df2dict(info)
    return info, origin_num


def df2dict(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    将 pandas Dataframe 转成两层嵌套的 dict
    并且将第 1 列作为第一层 dict 的 key，其他列名和列值分别作为第二层 dict 的 key 和 value

    :param df: 待转换的 Dataframe
    :return: 转换后的嵌套 dict
    """
    line_keys = df.columns[1:]
    df_dict = {}
    for line in df.values:
        if not df_dict.__contains__(line[0]):
            df_dict[line[0]] = dict(zip(line_keys, line[1:]))
    return df_dict


def save_prompt(prompts, save_path, save_sep):
    with open(save_path, 'w', encoding='utf-8') as file:
        print('prompt saved', save_path)
        for uid, prompt in prompts.items():
            file.write(f'{uid}{save_sep}{prompt}\r\n')


def save_user_mulit_prompt(prompts, save_path, save_sep):
    # 同一个user有多个prompt
    with open(save_path, 'w', encoding='utf-8') as file:
        print('multi prompt saved', save_path)
        for uid, prompt_list in prompts.items():
            for prompt in prompt_list:
                file.write(f'{uid}{save_sep}{prompt}\r\n')


def load_user_click_info(user_click_info_path):
    data_paths = os.listdir(user_click_info_path)
    if '__SUCCESS' in data_paths:
        data_paths.remove('__SUCCESS')
    if '_SUCCESS' in data_paths:
        data_paths.remove('_SUCCESS')
    if '.ipynb_checkpoints' in data_paths:
        data_paths.remove('.ipynb_checkpoints')
    user_click_info = None
    for _, path in enumerate(data_paths):
        cur_path = os.path.join(user_click_info_path, path)
        tmp_user_click_info = pd.read_orc(cur_path)
        if user_click_info is None:
            user_click_info = tmp_user_click_info
        else:
            user_click_info = pd.concat([user_click_info, tmp_user_click_info], axis=0)
    print("user_click_info shape:", user_click_info.shape)
    return user_click_info


def load_user_data_all(item_info, save_path, args):
    """
    :param dir_path: 用户样例的文件路径，包括用户长周期的完播、收藏、下载历史
    :param user_click_info_path: 用户训练集信息的文件路径，包含用户在新数据中的样本数量
    :param item_info: item的文字信息
    :param save_path: 存储路径
    :param save_sep: 存储分隔符
    :param user_count_train_threshold: 用户训练集样本数阈值
    :param user_count_his_threshold: 用户历史样本阈值
    :return:
    """
    dir_path = args.user_info_path
    user_click_info_path = args.user_click_info_path
    save_sep = args.sep
    user_count_train_threshold = args.user_count_train_threshold
    user_count_his_threshold = args.user_count_his_threshold
    max_user_seq_len = args.max_user_seq_len
    avg_user_seq_len = args.avg_user_seq_len

    data_paths = os.listdir(dir_path)
    if '__SUCCESS' in data_paths:
        data_paths.remove('__SUCCESS')
    if '_SUCCESS' in data_paths:
        data_paths.remove('_SUCCESS')
    if '.ipynb_checkpoints' in data_paths:
        data_paths.remove('.ipynb_checkpoints')
    all_info = {}
    total_num, valid_num, prompt_num = 0, 0, 0
    begin_time = time.time()
    # load user需要读入用户在训练集的样本数，过滤用户：训练集样本>=100，历史听歌正样本>=5
    user_click_info = load_user_click_info(user_click_info_path)
    user_click_info["click_cnt"] = user_click_info['click_cnt'].astype('float')
    for idx, path in enumerate(data_paths):
        cur_path = os.path.join(dir_path, path)
        print(f'load data from {cur_path} idx: {idx}')
        info = pd.read_orc(cur_path)
        print('load shape', info.shape)
        total_num += info.shape[0]
        info.replace('\\N', None, inplace=True)
        info = info[FULL_NECESSARY_USER_COLUMNS]
        info = info.loc[(info['play_song_set_90dy'].notnull()) |
                        (info['collect_song_set_90dy'].notnull()) |
                        (info['download_song_set_90dy'].notnull())]

        info = pd.merge(info, user_click_info, how="inner", on="user_id")
        info = info[info.click_cnt >= user_count_train_threshold]
        print('data after filter1 by history:', info.shape)
        info = df2dict(info)
        prompts, valid_prompt_num = format_user_template(info, item_info, user_count_his_threshold, max_user_seq_len,
                                                         avg_user_seq_len)
        print('data after filter2 user num:', len(prompts))
        valid_num += len(prompts)
        prompt_num += valid_prompt_num
        save_user_mulit_prompt(prompts, os.path.join(save_path, 'part_' + str(idx)), save_sep)
        print("curr file end cost time:", time.time() - begin_time)
    print('total user num:', total_num, 'valid user num:', valid_num, 'prompt_num:', prompt_num)

    for key in list(prompts.keys())[:3]:
        print('prompt example: ', prompts.get(key, "no prompt"))
    return all_info


def load_item_data_all(dir_path, columns, sep):
    data_paths = os.listdir(dir_path)
    if '__SUCCESS' in data_paths:
        data_paths.remove('__SUCCESS')
    if '_SUCCESS' in data_paths:
        data_paths.remove('_SUCCESS')
    if '.ipynb_checkpoints' in data_paths:
        data_paths.remove('.ipynb_checkpoints')
    all_info = {}
    total_num, valid_num = 0, 0
    print(f'data_path: {data_paths}')
    begin_time = time.time()
    for _, path in enumerate(data_paths):
        cur_path = os.path.join(dir_path, path)
        print("---------------------- load item data ------------------------------------------")
        info, num_per_file = load_item_data(cur_path, columns, sep, FULL_NECESSARY_ITEM_COLUMNS)
        total_num += num_per_file
        valid_num += len(info)
        all_info.update(info)
        print("curr file end cost time:", time.time() - begin_time)
    print('total item num:', total_num, 'valid item num:', valid_num)
    return all_info


def get_song_info(song_info_line: Dict[str, str]) -> Tuple[str, str, str, str, str]:
    song_name = song_info_line[SONG_NAME]
    artist_names = '、'.join(song_info_line[SONG_ARTIST_NAME_SET].strip().split(SONG_ARTIST_SEP))
    album_name = song_info_line[ALBUM_NAME]

    genres_tag_name = song_info_line[SONG_GENRES_TAG_NAME]
    langua_tag_name = song_info_line[SONG_LANGUA_TAG_NAME]
    song_info = [song_name, artist_names, album_name, genres_tag_name, langua_tag_name]
    return song_info


def get_songs_info_str(song_str_list: List[str], sep: str,
                       song_dict: Dict[str, Dict[str, str]], max_song: int = 0) -> Tuple[List[str], Set[str]]:
    song_info_list = []
    not_found_set = set([])
    song_list = []
    for song_str in song_str_list:
        if song_str is not None:
            song_list.extend(song_str.split(sep))
    for song_id in song_list:
        if song_id in song_dict:
            song_info = get_song_info(song_dict[song_id])
            song_name, artist_names, album_name, genres_tag_name, langua_tag_name = song_info
            song_info_list.append(f'《{song_name}》（{genres_tag_name}，{langua_tag_name}）')
        else:
            not_found_set.add(song_id)
    if max_song > 0:
        song_info_list = song_info_list[:max_song]
    return song_info_list, not_found_set


def format_user_template(user_info, item_info, user_count_his_threshold, max_user_seq_len, avg_user_seq_len):
    """
    生成user的prompt
    :param user_info: 用户浏览信息记录表
    :param item_info: 物品信息的记录表
    :param user_count_his_threshold: 用户历史样本阈值
    :param max_user_seq_len: 用于生成知识的最大用户序列长度；
    :param avg_user_seq_len: 用于生成每个prompt的用户序列长度；
    :return: user prompt
    """
    prompt = {}
    song_missing_set = set([])
    total_pos_song_num, total_neg_song_num, total_prompt_token_num = 0, 0, 0
    valid_prompt_num = 0
    user_with_song_less_than_five_num = 0
    user_num = 0
    max_prompt_token_num = 0
    token_num_large_than_768, token_num_large_than_1024 = 0, 0
    top_num_0_num = 0
    for info_key, info_values in user_info.items():
        user_age = AGE_MAPPING.get(info_values['age'], '年龄未知')
        user_city = info_values.get('province', '')
        if user_city is None:
            user_city = '城市未知'
        user_gender = GENDER_MAPPING.get(info_values['gender'], '性别未知的')
        sep = USER_SONG_SET_SEP

        like_song_strs = [info_values['download_song_set_90dy'], info_values['collect_song_set_90dy'],
                          info_values['play_song_set_90dy']]
        # 只保留history >= 10的用户，如果history>=20,切成多个，以####为间隔存在prompt
        # 下载放在最前，收藏其实，然后是完播
        user_like_songs, missing_1 = get_songs_info_str(like_song_strs, sep, item_info, max_user_seq_len)
        song_missing_set.update(missing_1)
        if len(user_like_songs) < user_count_his_threshold:
            user_with_song_less_than_five_num += 1
            continue
        user_num += 1

        while True:
            if len(user_like_songs) < user_count_his_threshold:
                break
            top_num = avg_user_seq_len
            user_like_song_str = "，".join(user_like_songs[:top_num])
            line = f"<|system|>\nYou are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's " \
                   f"instructions carefully. Respond using markdown.\n<|user|>\n你是一个推荐助手，你需要为用户提供个性化的分析。" \
                   f"一个{user_age}的常驻{user_city}的{user_gender}用户，根据他喜欢的歌曲列表：{user_like_song_str}。" \
                   f"分别从风格、语言、情感、节奏这些角度分析用户对歌曲的偏好（根据用户喜欢的歌曲列表给出清晰的解释）。" \
                   f"***注意每个角度的分析和解释不超过30个字***。\n<|assistant|>"
            while len(line) > 1024 and top_num > 0:
                top_num -= 1
                user_like_song_str = "，".join(user_like_songs[:top_num])
                line = f"<|system|>\nYou are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's " \
                       f"instructions carefully. Respond using markdown.\n<|user|>\n你是一个推荐助手，你需要为用户提供个性化的分析。" \
                       f"一个{user_age}的常驻{user_city}的{user_gender}用户，根据他喜欢的歌曲列表：{user_like_song_str}。" \
                       f"分别从风格、语言、情感、节奏这些角度分析用户对歌曲的偏好（根据用户喜欢的歌曲列表给出清晰的解释）。" \
                       f"***注意每个角度的分析和解释不超过30个字***。\n<|assistant|>"
            if top_num == 0:
                top_num_0_num += 1
                break
            total_prompt_token_num += len(line)
            if len(line) > max_prompt_token_num:
                max_prompt_token_num = len(line)
            if info_key not in prompt:
                prompt[info_key] = [line]
            else:
                prompt[info_key].append(line)
            valid_prompt_num += 1
            user_like_songs = user_like_songs[top_num:]

    print('song missing', len(song_missing_set))
    print('valid user:', user_num, "valid prompt:", valid_prompt_num)
    print(f'avg prompt token num: {total_prompt_token_num / (valid_prompt_num + 0.0001)}')
    print(f'token_num_large_than_768:{token_num_large_than_768}; token_num_large_than_1024:{token_num_large_than_1024}')
    print(f'max prompt token num: {max_prompt_token_num}')
    print(f'top_num_0_num: {top_num_0_num}')
    return prompt, valid_prompt_num


def generate_prompt(args):
    user_info_path = args.user_info_path
    item_info_path = args.item_info_path
    item_columns = args.item_columns
    save_path = args.save_path
    user_prefix = args.user_prefix
    print("user_info_path:", user_info_path, "item_info_path:", item_info_path)
    item_info = load_item_data_all(item_info_path, item_columns, FULL_ITEM_COLUMN_SEP)
    user_save_path = os.path.join(save_path, user_prefix)
    print("user_save_path:", user_save_path)
    os.makedirs(user_save_path, exist_ok=True)

    _ = load_user_data_all(item_info, user_save_path, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user_info_path', required=True, type=str, help='Please specify the user info path')
    parser.add_argument('-uc', '--user_click_info_path', required=True, type=str, help='Please specify the user click'
                                                                                       ' info path')
    parser.add_argument('-i', '--item_info_path', required=True, type=str, help='Please specify the item info path')
    parser.add_argument('-ci', '--item_columns', required=False, default='',
                        help='Please specify the item column names')
    parser.add_argument('-su', '--sep', required=False, type=str, default='|')
    parser.add_argument('-up', '--user_prefix', required=False, default='sema_user_like', type=str,
                        help='Prefix for saving user files')
    parser.add_argument('-m', '--max_user_seq_len', required=False, default=40, type=int)
    parser.add_argument('-a', '--avg_user_seq_len', required=False, default=20, type=int)
    parser.add_argument('-p', '--save_path', required=False, default='')
    parser.add_argument('-t1', '--user_count_train_threshold', required=False, default=100, type=int)
    parser.add_argument('-t2', '--user_count_his_threshold', required=False, default=5, type=int)

    args, unknown = parser.parse_known_args()
    print('args:', args)
    print('unknown arguments: ', unknown)
    if not args.item_columns:
        args.item_columns = FULL_ITEM_COLUMNS
    else:
        args.item_columns = args.item_columns.split(',')

    print('args', args)
    seed = 1234
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    generate_prompt(args)


if __name__ == '__main__':
    main()
