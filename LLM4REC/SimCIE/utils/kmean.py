import argparse
import os
import stat
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def write_to_file(save_file, mode='w'):
    flags = os.O_WRONLY | os.O_CREAT
    stats = stat.S_IWUSR | stat.S_IRUSR
    file_hander = os.fdopen(os.open(save_file, flags, stats), mode)
    return file_hander


def norm_embedding(data_file, new_data_file):
    data = pd.read_csv(data_file, sep="|", names=["item_id", "item_embedding"])  # item_ids,item_embeddings
    print(data.head(5))

    def get_norm_embed(s):
        emb = np.fromstring(s, sep=",", dtype=float)
        return emb / np.linalg.norm(emb)

    data["norm_embedding"] = data["item_embedding"].apply(get_norm_embed)
    output = write_to_file(new_data_file, "w")
    for item_id, item_embed in zip(data["item_id"], data["norm_embedding"]):
        embedding = ",".join(map(str, item_embed.tolist()))
        output.write(str(item_id) + "|" + str(embedding) + "\n")


def cluster_items(item_embeddings_file, new_item_file, n_clusters):
    item_embeddings_df = pd.read_csv(item_embeddings_file, sep="|", names=["item_id", "item_embedding"])
    print("length of item_embeddings_df", len(item_embeddings_df))
    print("the dataframe of item embeddings")
    print(item_embeddings_df.head(5))

    embed_list = item_embeddings_df["item_embedding"].str.split(",").tolist()
    emb_list_array = np.array([np.array(list(map(float, v))) for v in embed_list])
    cluster_kmeans_y = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters).fit_predict(emb_list_array)
    new_item_df = item_embeddings_df.assign(cluster_id=cluster_kmeans_y)
    print("length of item with cluster_id", len(new_item_df))
    print("the dataframe of item with cluster_id")
    print(new_item_df.head(5))
    new_item_df.to_csv(new_item_file, index=False, header=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrained_item_embedding_file', required=True, type=str)
    parser.add_argument('-d', '--data_path', required=True, type=str)
    parser.add_argument('-n', '--norm', required=False, type=str, default="false")
    parser.add_argument('-c', '--n_clusters', required=False, type=int, default=500)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    start_time = time.time()

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
        print("make dir: ", args.data_path)

    new_item_file = os.path.join(args.data_path, "kmean_embedding.csv")

    if args.norm.lower() == "true":
        norm_embedding_file = os.path.join(args.data_path, "norm_pretrained_embedding.txt")
        norm_embedding(args.pretrained_item_embedding_file, norm_embedding_file)
        cluster_items(norm_embedding_file, new_item_file, args.n_clusters)
    else:
        cluster_items(args.pretrained_item_embedding_file, new_item_file, args.n_clusters)

    end_time = time.time()
    print(f"time: {end_time - start_time}")


if __name__ == '__main__':
    main()
