'''
通过prompt_config_file配置文件，为不同领域的item定制不同的prompt
'''

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


def _construct_prompt(row, prompt_config):
    '''
    基于item prompt配置，构造prompt
    :param row: item基础字段信息
    :param prompt_config: item prompt配置
    :return: 单个item prompt
    '''
    # 领域下再细分item类型，比如阅读中的小说，听书等
    item_type = prompt_config.get("item_type", None)
    default_prompt = prompt_config.get("default_prompt", "")
    fields_length = prompt_config.get("fields_length", {})

    if item_type is not None:
        prompt_str = prompt_config.get("prompt_str", {})
        item_type_value = row[item_type]
        prompt_template = prompt_str.get(item_type_value, "")
    else:
        prompt_template = default_prompt

    prompt = prompt_template
    for field, length in fields_length.items():
        value = str(row[field])[:int(length)]
        prompt = prompt.replace(field, value)
    prompt = prompt.replace("None", '')
    return prompt


def construct_domain_prompt(domain_item_df, domain_item_prompt_config, domain_field):
    '''
    为单个领域的item构造prompt
    :param domain_item_df: 领域item数据
    :param domain_item_prompt_config:领域item prompt配置
    :param domain_field: 领域名
    :return: 对应领域的item prompt df
    '''
    # 获取表中item_id对应的字段名
    item_id_col = domain_item_prompt_config.get("id_col", None)

    # 获取prompt中包含的所有字段
    info_columns = domain_item_prompt_config.get("info_columns", None)
    if item_id_col not in info_columns:
        info_columns.append(item_id_col)

    if info_columns is not None:
        info_df = pd.concat([domain_item_df[v] for v in info_columns], axis=1)
    else:
        info_df = domain_item_df

    # 删除配置字段为空的行数据
    dropna_columns = domain_item_prompt_config.get("dropna_columns", None)
    if dropna_columns is not None:
        info_df.dropna(subset=dropna_columns, inplace=True)
    info_df.fillna("未知", inplace=True)

    # 构造item prompt
    prompt_configs = domain_item_prompt_config.get("prompts")
    domain_item_df["info"] = domain_item_df.apply(_construct_prompt, args=(prompt_configs,), axis=1)

    # 仅保留item domain，item id， item prompt
    prompt_df = pd.concat([domain_item_df[domain_field], domain_item_df[item_id_col], domain_item_df["info"]], axis=1)
    prompt_df.rename(columns={item_id_col: "item_id"}, inplace=True)
    prompt_df.dropna(subset=["item_id", "info"], inplace=True)
    print("item prompt df")
    print(len(prompt_df))
    print(prompt_df.tail(5))
    return prompt_df


def get_domain_item_prompt(item_data_path, item_prompt_file, fields_config):
    '''
    获取各个领域的item prompt
    :param item_data_path: item基础信息表数据路径，包含多个领域
    :param item_prompt_file: 输出item prompt文件
    :param fields_config: item prompt配置信息
    :return: None
    '''
    domain_config = fields_config.get("domain_config", None)
    item_prompt_config = fields_config.get("item_prompt_config")

    total_prompt_df_list = []
    for _, file in enumerate(os.listdir(item_data_path)):
        if file.startswith("_"):
            continue

        item_file = os.path.join(item_data_path, file)
        print("item_file", item_file)
        item_df = pd.read_orc(item_file)

        if domain_config is not None:
            domain_field = domain_config.get("domain_field")
            domain_values = domain_config.get("domain_values")

            multi_domain_df_list = []
            domain_item_df = item_df.groupby(domain_field)
            for domain, group in domain_item_df:
                if domain not in domain_values:
                    continue
                print(domain)
                domain_item_prompt_config = item_prompt_config.get(domain)
                domain_prompt_df = construct_domain_prompt(group, domain_item_prompt_config, domain_field)
                multi_domain_df_list.append(domain_prompt_df)
            prompt_df = pd.concat(multi_domain_df_list, axis=0)
        else:
            item_prompt_config = fields_config.get("item_prompt_config")
            domain_item_prompt_config = item_prompt_config.get("domain")
            prompt_df = construct_domain_prompt(item_df, domain_item_prompt_config, "domain")
        total_prompt_df_list.append(prompt_df)

    total_prompt_df = pd.concat(total_prompt_df_list, axis=0)
    print("item prompt df")
    print(len(total_prompt_df))
    print(total_prompt_df.columns)
    print(total_prompt_df.head(5))
    prompt_output = write_to_file(item_prompt_file, 'w')
    total_prompt_df.to_csv(prompt_output, index=False, header=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--item_file_path', required=True, type=str)
    parser.add_argument('-o', '--item_prompt_path', required=True, type=str)
    parser.add_argument('-c', '--prompt_config_file', required=True, type=str)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    with open(args.prompt_config_file, 'r', encoding='utf-8') as fin:
        fields_config = json.load(fin)

    # 加载mtp平台配置文件
    mtp_train_config_file = "/opt/huawei/schedule-train/algorithm/train.config"
    if os.path.exists(mtp_train_config_file):
        with open(mtp_train_config_file, 'r', encoding='utf-8') as fin:
            train_config = json.load(fin)
        fields_config = train_config
    print(f"fields_config: {fields_config}")

    start_time = time.time()

    if not os.path.exists(args.item_prompt_path):
        os.mkdir(args.item_prompt_path)

    item_prompt_file = os.path.join(args.item_prompt_path, "item_prompt.csv")
    get_domain_item_prompt(args.item_file_path, item_prompt_file, fields_config)

    end_time = time.time()
    print(f"time: {end_time - start_time}")


if __name__ == '__main__':
    main()
