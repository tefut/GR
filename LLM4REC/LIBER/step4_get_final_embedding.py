import argparse
import copy
import json
import os
from io import open

import numpy as np
import pandas as pd
from constant import FULL_ITEM_COLUMNS, FULL_ITEM_COLUMN_SEP, USER_SONG_SET_SEP, SONG_TAG_SEP
from sklearn.cluster import MiniBatchKMeans
from step1_music_prompt_generator import load_item_data_all


def get_final_embedding(in_path, embedding_dim, sep):
    path = in_path + "/encoding_dim_%s" % embedding_dim
    file_names = os.listdir(path)
    user_emb_dict = dict()
    for file in file_names:
        source_file = os.path.join(path, file)
        lines = open(source_file, 'r', encoding='utf-8', newline='\n').readlines()
        for line in lines:
            uid, emb = line.rstrip('\n').split(sep)
            emb = json.loads("[" + emb + "]")
            if uid in user_emb_dict:
                user_emb_dict[uid] = user_emb_dict[uid] + emb
            else:
                user_emb_dict[uid] = emb
    liber_user_emb_dict = dict()
    for uid in user_emb_dict:
        if len(user_emb_dict[uid]) == embedding_dim:
            liber_user_emb_dict[uid] = np.array(copy.deepcopy(user_emb_dict[uid]))
        else:
            tmp_emb = np.array(copy.deepcopy(user_emb_dict[uid]))
            tmp_emb = np.reshape(tmp_emb, (-1, embedding_dim))
            tmp_emb = np.average(np.reshape(tmp_emb, (-1, embedding_dim)), axis=0)
            liber_user_emb_dict[uid] = tmp_emb
    liber_user_emb_dict['xx'] = np.average(np.array(list(liber_user_emb_dict.values())), axis=0)
    print("len liber_user_emb_dict:", len(liber_user_emb_dict))
    save_path2 = in_path + "/final_encoding_dim_%s_liber" % embedding_dim
    num = 0
    with open(save_path2, 'w', encoding='utf-8') as output:
        for uid in liber_user_emb_dict:
            tmp_list = liber_user_emb_dict[uid]
            if tmp_list.shape[0] != embedding_dim:
                raise ValueError("Error")
            embed_txt = ','.join(list(map(lambda x: str(round(x, 8)), tmp_list)))
            output_str = uid + '|' + embed_txt
            output.write(output_str + '\n')
            num += 1
    print("save user num:", num)
    return liber_user_emb_dict


def cluster_llm_user(llm_user_emb_dict, n_clusters=500):
    uid_list = []
    user_emb_list = []
    for uid in llm_user_emb_dict:
        uid_list.append(uid)
        user_emb_list.append(llm_user_emb_dict[uid])
    uid_list_array = np.array(uid_list)
    user_emb_list_array = np.array(user_emb_list)
    user_cluster_kmeans_y = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters).fit_predict(user_emb_list_array)
    print(f'kmeans end')
    llm_user_cluster_dict = dict()
    for i in range(uid_list_array.shape[0]):
        tmp_uid, tmp_cluster = uid_list_array[i], user_cluster_kmeans_y[i]
        llm_user_cluster_dict[tmp_uid] = tmp_cluster
    print('cluster_num:', n_clusters)
    return llm_user_cluster_dict, n_clusters


def top_play_most_tag(x, top_num):
    x_list = sorted(x[:top_num])
    x_list_str = ','.join(x_list)
    return x_list_str


def cluster_other_user(user_path, item_path, threshold=200):
    item_info_dict = load_item_data_all(item_path, FULL_ITEM_COLUMNS, FULL_ITEM_COLUMN_SEP)

    def get_three_most_tag(x):
        song_list = x.split(USER_SONG_SET_SEP)
        user_like_tag = dict()
        for song_id in song_list:
            song_info = item_info_dict.get(song_id, None)
            if song_info is not None:
                genres_tag_name = song_info["genres_tag_name"].split(SONG_TAG_SEP)
                for genres_tag in genres_tag_name:
                    tmp_tag = genres_tag.strip()
                    if tmp_tag:
                        if tmp_tag in user_like_tag:
                            user_like_tag[tmp_tag] += 1
                        else:
                            user_like_tag[tmp_tag] = 1
        temp_list = sorted(user_like_tag.items(), key=lambda item: item[1], reverse=True)
        user_like_tag_top3 = [temp_list[i][0] for i in range(min(3, len(temp_list)))]
        return user_like_tag_top3

    data_paths = os.listdir(user_path)
    if '__SUCCESS' in data_paths:
        data_paths.remove('__SUCCESS')
    if '_SUCCESS' in data_paths:
        data_paths.remove('_SUCCESS')
    if '.ipynb_checkpoints' in data_paths:
        data_paths.remove('.ipynb_checkpoints')
    final_info = None
    for idx, path in enumerate(data_paths):
        cur_path = os.path.join(user_path, path)
        print(f'load data from {cur_path} idx: {idx}')
        info = pd.read_orc(cur_path)
        info['play_most_tag'] = info['play_song_set_90dy'].apply(get_three_most_tag)
        new_info = info[['user_id', 'play_most_tag']].copy(deep=True)
        if final_info is None:
            final_info = new_info
        else:
            final_info = pd.concat([new_info, final_info])
        print("idx:", idx, info.shape, new_info.shape, final_info.shape)

    final_info['cluster_identifier'] = final_info['play_most_tag'].apply(top_play_most_tag, top_num=3)

    cluster_identifier_value_counts = final_info['cluster_identifier'].value_counts()

    cluster_identifier_to_cluster_dict = dict()
    tmp_num = 0
    for key in cluster_identifier_value_counts.keys():
        if cluster_identifier_value_counts[key] >= threshold:
            cluster_identifier_to_cluster_dict[key] = tmp_num
            tmp_num += 1

    print("cluster num:", len(cluster_identifier_value_counts), "remain cluster_other_user num:", tmp_num)

    other_user_cluster_dict = dict()
    for _, row in final_info.iterrows():
        tmp_uid, tmp_cluster_identifier = row['user_id'], row['cluster_identifier']
        tmp_cluster = cluster_identifier_to_cluster_dict.get(tmp_cluster_identifier, None)
        if tmp_cluster is not None:
            other_user_cluster_dict[tmp_uid] = tmp_cluster

    print("user num:", final_info.shape[0], "remain user num:", len(other_user_cluster_dict))
    return other_user_cluster_dict, tmp_num


def cluster_all_user(user_info_path, item_info_path, llm_user_emb_dict, llm_input_path, llm_cluster=500,
                     other_threshold=200):
    llm_user_cluster_dict, llm_cluster_num = cluster_llm_user(llm_user_emb_dict, n_clusters=llm_cluster)
    other_user_cluster_dict, other_cluster_num = cluster_other_user(user_info_path, item_info_path,
                                                                    threshold=other_threshold)
    print("llm_user_cluster_dict:", len(llm_user_cluster_dict), "llm_cluster_num:", llm_cluster_num)
    print("other_user_cluster_dict:", len(other_user_cluster_dict), "other_cluster_num:", other_cluster_num)
    merge_user_cluster_dict = dict()
    tmp_num1, tmp_num2 = 0, 0
    for tmp_uid in llm_user_cluster_dict:
        merge_user_cluster_dict[tmp_uid] = llm_user_cluster_dict[tmp_uid] + 2
        tmp_num2 += 1
    for tmp_uid in other_user_cluster_dict:
        if tmp_uid not in llm_user_cluster_dict:
            merge_user_cluster_dict[tmp_uid] = other_user_cluster_dict[tmp_uid] + llm_cluster_num + 6
            tmp_num1 += 1

    print("merge_user_cluster_dict:", len(merge_user_cluster_dict))
    print("tmp_num1:", tmp_num1, "tmp_num2:", tmp_num2)
    all_cluster_num = max(merge_user_cluster_dict.values()) + 1
    print("all_cluster_num:", all_cluster_num)
    save_path = llm_input_path + "/user_group_id"
    with open(save_path, 'w', encoding='utf-8') as output:
        for tmp_uid in merge_user_cluster_dict:
            tmp_cluster_id = merge_user_cluster_dict[tmp_uid]
            output_str = str(tmp_uid) + '|' + str(tmp_cluster_id)
            output.write(output_str + '\n')
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user_info_path', required=True, type=str, help='Please specify the user info path')
    parser.add_argument('-i', '--item_info_path', required=True, type=str, help='Please specify the item info path')
    parser.add_argument('-lp', '--llm_input_path', required=True, type=str, help='Please specify input file path')
    parser.add_argument('-pt', '--pca_target_dim', required=False, type=int, default=64)
    parser.add_argument('-su', '--sep', required=False, type=str, default='|')
    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    llm_user_emb_dict = get_final_embedding(args.llm_input_path, args.pca_target_dim, args.sep)
    cluster_all_user(args.user_info_path, args.item_info_path, llm_user_emb_dict, args.llm_input_path)


if __name__ == '__main__':
    main()
