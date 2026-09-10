import argparse
import os
import time
from io import open
from typing import List

import numpy as np
from incremental_pca import reduce_dimensionality, batch_read_data
from knowledge_generator import LLMKG
from sklearn.decomposition import IncrementalPCA


def generate_embedding(in_path: str, out_path: str = "",
                       file_names: List[str] = None,
                       llm_batch_size: int = 512,
                       pca_batch_size: int = 10000,
                       target_dim: int = -1):
    remains = []
    encoding_dim = 0
    ipca = IncrementalPCA(n_components=target_dim, batch_size=pca_batch_size) if target_dim > 0 else None
    for tmp_file_name in file_names:
        print("file_name:", tmp_file_name)
        source_file = os.path.join(in_path, tmp_file_name)
        lines = open(source_file, 'r', encoding='utf-8', newline='\r\n').readlines()
        raw_data = []
        tmp_i = 0
        for line in lines:
            uid, response = line.rstrip('\r\n').split(args.sep)
            response_list = response.split("你是一个推荐助手，你需要为用户提供个性化的分析。")
            if len(response_list) >= 1:
                response2 = response_list[1].strip()
            else:
                response2 = response.strip()
            raw_data.append([uid, response2])
            tmp_i += 1
            if tmp_i == 1:
                print("encode:", response2)

        data = np.array(raw_data)
        print("data:", data.shape)
        print("data example:", data[0])
        t = time.time()
        res = llm_kg.encode_knowledge(data[:, -1], llm_batch_size)
        encoding_dim = len(res[0])
        total_time = time.time() - t
        print(f'data num:{len(data)}, '
              f'data dim {len(res[0])}, '
              f'total time: {total_time}, '
              f'time per sample: {total_time / len(data)}')
        if ipca is not None:
            for _, batch_emb in batch_read_data(np.array(res), batch_size=pca_batch_size):
                if batch_emb.shape[0] == pca_batch_size:
                    ipca.partial_fit(batch_emb)
                else:
                    remains.extend(batch_emb)
                    if len(remains) > pca_batch_size:
                        ipca.partial_fit(remains[:pca_batch_size])
                        remains = remains[pca_batch_size:]
            print('incremental pca for encoding')

        target_path = os.path.join(out_path, 'encoding_dim_' + str(encoding_dim))
        os.makedirs(target_path, exist_ok=True)
        target_file = os.path.join(target_path, tmp_file_name)
        with open(target_file, 'w', encoding='utf-8') as output:
            for uid, embed in zip(data[:, 0], res):
                embed_txt = ','.join(list(map(str, embed)))
                output.write(uid + args.sep + embed_txt + '\n')
        print('encoding saving in', target_file)
    return ipca, encoding_dim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True, type=str,
                        help='Please specify pretrained language model path')
    parser.add_argument('-t', '--model_type', required=False, default='MiniLM-L12-v2',
                        help='Please specify model type')
    parser.add_argument('-i', '--input_path', required=True, type=str,
                        help='Please specify input file path')
    parser.add_argument('-o', '--output_path', required=False, default='',
                        help='Please specify output file path')
    parser.add_argument('-lb', '--llm_batch_size', required=False, type=int,
                        default=512, help='batch size for llm encoding')
    parser.add_argument('-pb', '--pca_batch_size', required=False, type=int,
                        default=10000, help='batch size for pca')
    parser.add_argument('-pt', '--pca_target_dim', required=False, type=int,
                        default=64, help='target dimension for pca')
    parser.add_argument('-s', '--sep', required=False, default='|',
                        help='Please specify the seperator for input and output')
    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    llm_kg = LLMKG(os.path.join(args.model_path, args.model_type))

    print('input path', args.input_path)
    print('output_path', args.output_path)
    # 多个文件推理
    os.makedirs(args.output_path, exist_ok=True)
    target_file_names = os.listdir(args.input_path)
    pca1, source_dim = generate_embedding(args.input_path, args.output_path, target_file_names,
                                          args.llm_batch_size, args.pca_batch_size, args.pca_target_dim)
    source_file_path = os.path.join(args.output_path, "encoding_dim_" + str(source_dim))

    target_file_path = os.path.join(args.output_path, "encoding_dim_" + str(args.pca_target_dim))
    os.makedirs(target_file_path, exist_ok=True)
    for name in target_file_names:
        reduce_dimensionality(pca1, os.path.join(source_file_path, name), args.pca_batch_size,
                              os.path.join(target_file_path, name), args.sep)
