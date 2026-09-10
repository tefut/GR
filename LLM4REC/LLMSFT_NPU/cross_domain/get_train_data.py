'''构建 LLM SFT跨域训练数据，流程如下：
步骤一：基于正样本对取其中的item，将business和item_id进行拼接，作为索引。将item1和item2进行concat，去重，重建索引；
步骤二：将步骤一的结果和item embedding取交集；
步骤三：将步骤二的结果和item_info取交集；
步骤四：基于交集数据，构造index_to_item.csv, item_embedding.pickle, predict.csv;
步骤五：去除样本对中，item1或item2不在 index_to_item中的样例；
步骤六：基于步骤五和predict.csv，构造 train.csv
'''

import argparse
import os
import pickle
import stat

import pandas as pd
import torch


def write_to_file(save_file, mode='w'):
    flags = os.O_WRONLY | os.O_CREAT
    stats = stat.S_IWUSR | stat.S_IRUSR
    file_hander = os.fdopen(os.open(save_file, flags, stats), mode)
    return file_hander


def load_data_from_file(data_file, sep=",", names=None):
    if data_file.endswith("csv"):
        data_df = pd.read_csv(data_file, names=names, sep=sep, encoding="utf-8")
    else:
        data_df = pd.read_orc(data_file)
    return data_df


def load_data(data_path, sep=None, names=None):
    data_df_list = []

    if os.path.isfile(data_path):
        data_df_list.append(load_data_from_file(data_path, sep, names))
    else:
        for file in os.listdir(data_path):
            if file == "_SUCCESS" or file.endswith("json") or file.startswith("."):
                continue

            file_path = os.path.join(data_path, file)
            print(file_path)
            data_df = load_data_from_file(file_path)
            data_df_list.append(data_df)

    total_data_df = pd.concat(data_df_list, axis=0)
    return total_data_df


def load_item_sample_df(data_path):
    '''
    加载样本对，提取样本对中包含的所有item，并去重；
    :param data_path: 样本对数据路径
    :return: 样本对dataframe和样本对中去重的items
    '''
    data_df_list = []
    for file in os.listdir(data_path):
        if file == "_SUCCESS" or file.endswith("json") or file.startswith("."):
            continue

        file_path = os.path.join(data_path, file)
        print(file_path)

        if file.endswith("csv"):
            data_df = pd.read_csv(file_path, encoding="utf-8")
        else:
            data_df = pd.read_orc(file_path)

        data_df["domain_item_1"] = data_df['busine_type_1'].str.cat(data_df["item_id_1"], sep="_")
        data_df["domain_item_2"] = data_df['busine_type_2'].str.cat(data_df["item_id_2"], sep="_")
        new_data_df = pd.concat([data_df["domain_item_1"], data_df["domain_item_2"]], axis=1)
        data_df_list.append(new_data_df)

    total_data_df = pd.concat(data_df_list, axis=0)
    print("item samples df")
    print(len(total_data_df))
    print(total_data_df.head(5))

    col1 = total_data_df["domain_item_1"].drop_duplicates()
    col2 = total_data_df["domain_item_2"].drop_duplicates()
    col = pd.concat([col1, col2])
    col = col.drop_duplicates()
    item_df = pd.DataFrame(col, columns=['domain_item'])
    print(f"the number of items: {len(item_df)}")
    print(item_df.head(5))

    return total_data_df, item_df


def add_embedding_and_prompt(item_sample_path, item_embedding_path, item_prompt_path):
    '''
    加载预训练item embedding文件和item prompt文件，为训练数据中item增加预训练item embedding和item prompt信息；
    :param item_sample_path: 正样本对文件路径
    :param item_embedding_path: 预训练item embedding文件路径
    :param item_prompt_path: item prompt文件路径
    :return: 带有预训练item embedding和item prompt的items df， 样本对 df
    '''

    # 加载训练正样本对，以及训练数据中的所有items
    item_sample_df, item_df = load_item_sample_df(item_sample_path)

    # 加载预训练item embedding文件
    item_embed_df = load_data(item_embedding_path, sep="|",
                              names=['item', 'business', 'action', 'ori_item_str', 'embedding'])
    item_embed_df["domain_item"] = item_embed_df["business"].str.cat(item_embed_df["item"], sep="_")

    # 为训练数据中的items 增加 预训练item embedding
    new_embed_df = pd.merge(item_df, item_embed_df, on="domain_item", how="left")
    new_embed_df = new_embed_df.dropna()
    new_embed_df = new_embed_df.drop(['item', 'business', 'action', 'ori_item_str'], axis=1)
    print("items with item embedding df")
    print(len(new_embed_df))
    print(new_embed_df.head(5))

    # 加载item prompt文件
    item_prompt_df = load_data(item_prompt_path)
    item_prompt_df["domain_item"] = item_prompt_df["business"].str.cat(item_prompt_df["item_id"], sep="_")

    # 为训练数据中的items 增加 item prompt信息
    new_item_df = pd.merge(new_embed_df, item_prompt_df, on="domain_item", how="left")
    new_item_df = new_item_df.dropna()
    new_item_df = new_item_df.drop(['item_id', "business"], axis=1)
    new_item_df["cluster_id"] = 0
    new_item_df = new_item_df.reset_index(drop=True)
    new_item_df = new_item_df.reset_index()
    print("items with item prompt df")
    print(len(new_item_df))
    print(new_item_df.head(5))

    return new_item_df, item_sample_df


def get_predict_item(item_info_df, index_to_item_file, item_embedding_file, predict_file):
    '''
    构建模型训练所需文件以及预测文件，包括item索引数据，item embedding压缩数据和模型预测数据
    :param item_info_df: 带有预训练item embedding和item prompt的items df
    :param index_to_item_file: 输出item索引文件
    :param item_embedding_file: 输出重建索引的embedding文件
    :param predict_file: 输出预测文件
    :return: None
    '''
    # 构建index_to_itemid文件
    index_to_item_df = pd.concat([item_info_df['index'], item_info_df['domain_item']], axis=1)
    print("index_to_itemid文件")
    print("length of index_to_item_df", len(index_to_item_df))
    print(index_to_item_df.head(5))
    index_to_item_df.to_csv(index_to_item_file, index=False, header=True, sep="|")

    # 构建item embedding pickle文件
    item_embedding_series = item_info_df.pop("embedding").apply(lambda x: x.split(","))
    item_embedding_list = item_embedding_series.apply(lambda x: [float(i) for i in x]).tolist()
    item_embedding_output = write_to_file(item_embedding_file, 'wb')
    pickle.dump(torch.Tensor(item_embedding_list).float(), item_embedding_output)

    # 构建模型预测数据
    new_columns = [item_info_df['index'], item_info_df['domain_item'], item_info_df['info'], item_info_df['cluster_id']]
    predict_data_df = pd.concat(new_columns, axis=1)
    print("模型预测数据")
    print("length of total item", len(predict_data_df))
    print(predict_data_df.head(5))
    predict_data_df.to_csv(predict_file, index=False)


def get_train_data(item_sample_df, new_item_df, train_file, eval_file):
    '''
    构建模型训练数据
    :param item_sample_df: item样本对 df
    :param new_item_df: 带有预训练item embedding和item prompt的items df
    :param train_file: 输出模型训练数据文件
    :param eval_file: 输出模型评估数据文件
    :return: None
    '''
    # 为样本对中的item添加预训练item embedding和item prompt
    left_item_info = pd.merge(item_sample_df, new_item_df, left_on="domain_item_1", right_on="domain_item", how="left")
    right_item_info = pd.merge(left_item_info, new_item_df, left_on="domain_item_2", right_on="domain_item", how="left")
    right_item_info.dropna(subset=["info_x", "info_y"], inplace=True)

    # 重建dataframe的columns，避免数据错乱
    new_columns_name = ["index_x", "domain_item_x", "info_x", "cluster_id_x", "index_y", "domain_item_y", "info_y",
                        "cluster_id_y"]
    new_columns = []
    for col in new_columns_name:
        new_columns.append(right_item_info[col])
    data_df = pd.concat(new_columns, axis=1)
    print("模型训练数据")
    print("the length of train data", len(data_df))
    print(data_df.head(15))
    data_df.to_csv(train_file, index=False)

    # 构建模型评估数据
    data_df.head(1000).to_csv(eval_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--item_sample_path', required=True, type=str)
    parser.add_argument('-e', '--item_embedding_path', required=True, type=str)
    parser.add_argument('-p', '--item_prompt_path', required=True, type=str)
    parser.add_argument('-o', '--save_path', required=True, type=str)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    item_sample_path = args.item_sample_path
    item_embedding_path = args.item_embedding_path
    item_prompt_path = args.item_prompt_path
    save_path = args.save_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    index_to_item_file = os.path.join(save_path, "index_to_item.csv")
    item_embedding_file = os.path.join(save_path, "item_embedding.pickle")
    predict_file = os.path.join(save_path, "predict.csv")
    train_file = os.path.join(save_path, "train.csv")
    eval_file = os.path.join(save_path, "eval.csv")
    new_item_df, item_sample_df = add_embedding_and_prompt(item_sample_path, item_embedding_path, item_prompt_path)
    get_predict_item(new_item_df, index_to_item_file, item_embedding_file, predict_file)
    get_train_data(item_sample_df, new_item_df, train_file, eval_file)


if __name__ == "__main__":
    main()
