# Created by xiyunjia on 2023/8/30
import argparse
import json
import os
from io import open

import pandas as pd
import torch
from incremental_pca import get_incremental_pca, reduce_dimensionality
from knowledge_generator import LLMKG, ChatGLMKG, Mistral
from tqdm import tqdm


def batch_read_data(data_file, batch_size, sep):
    df = pd.read_csv(data_file, sep=sep, names=["id", "prompt"])
    id_list = df['id'].values.tolist()
    prompt_list = df['prompt'].values.tolist()

    for i in range(0, len(id_list), batch_size):
        yield id_list[i:i + batch_size], prompt_list[i:i + batch_size]


def generator_fun(args):
    if args.model_type in ['chatglm-6b', 'chatglm-v2', 'chatglm-v3']:
        llm_kg = ChatGLMKG(os.path.join(args.model_path, args.model_type), max_length=args.max_length)
    else:
        llm_kg = LLMKG(os.path.join(args.model_path, args.model_type), max_length=args.max_length)

    sep = args.sep
    input_path = args.input_path
    output_path = args.output_path
    llm_batch_size = args.llm_batch_size

    os.makedirs(output_path, exist_ok=True)
    generate_path = os.path.join(output_path, 'generate')
    os.makedirs(generate_path, exist_ok=True)

    for _file_name in os.listdir(input_path):
        source_file = os.path.join(input_path, _file_name)
        generate_file = os.path.join(generate_path, _file_name)
        print(f"processing file: {source_file}")

        output = open(generate_file, 'w', encoding='utf-8')

        count = 0
        with torch.no_grad():
            for id_data, prompt_data in tqdm(batch_read_data(source_file, llm_batch_size, sep)):
                res = llm_kg.batch_generate(prompt_data)

                for _id, knowledge in zip(id_data, res):
                    output.write(_id + args.sep + knowledge + '\n')

                count += len(res)
                if count > 100000:
                    output.flush()
        print(f'generate knowlege saving in {generate_file}, the length of data: {count}')
        output.close()


def encode_fun(args):
    sep = args.sep
    input_path = args.input_path
    output_path = args.output_path
    llm_batch_size = args.llm_batch_size
    precision_float_number = args.precision_float_number

    if args.model_type in ['chatglm-6b', 'chatglm-v2', 'chatglm-v3']:
        llm_kg = ChatGLMKG(os.path.join(args.model_path, args.model_type), max_length=args.max_length)
    elif args.model_type in ['e5-mistral-7b-instruct']:
        llm_kg = Mistral(os.path.join(args.model_path, args.model_type), max_length=args.max_length)
    else:
        llm_kg = LLMKG(os.path.join(args.model_path, args.model_type), max_length=args.max_length)

    # 多个文件推理
    os.makedirs(output_path, exist_ok=True)
    encode_path = os.path.join(output_path, 'encode')
    os.makedirs(encode_path, exist_ok=True)

    for _file_name in os.listdir(input_path):
        source_file = os.path.join(input_path, _file_name)
        encode_file = os.path.join(encode_path, _file_name)
        print(f"processing file: {source_file}")

        output = open(encode_file, 'w', encoding='utf-8')

        count = 0
        with torch.no_grad():
            for id_data, prompt_data in tqdm(batch_read_data(source_file, llm_batch_size, sep)):
                res = llm_kg.encode_knowledge(prompt_data)

                if precision_float_number < 16:
                    res = [[round(v, precision_float_number) for v in _] for _ in res]

                for _id, embed in zip(id_data, res):
                    embed_txt = ','.join(list(map(str, embed)))
                    output.write(_id + args.sep + embed_txt + '\n')

                count += len(res)
                if count > 100000:
                    output.flush()
        print(f'encoding saving in {encode_file}, the length of data: {count}')
        output.close()


def pca_fun(args):
    pca_batch_size = args.pca_batch_size
    target_dim = args.pca_target_dim

    output_path = args.output_path
    encode_path = os.path.join(output_path, 'encode')
    pca_path = os.path.join(output_path, 'pca')
    os.makedirs(pca_path, exist_ok=True)

    source_files, target_files = [], []
    for file in sorted(os.listdir(encode_path)):
        source_files.append(os.path.join(encode_path, file))
        target_files.append(os.path.join(pca_path, file))

    pca_save_path = os.path.join(pca_path, 'pca' + '.pkl')

    pca = get_incremental_pca(source_files, pca_batch_size, target_dim, pca_save_path)
    for source_path, target_path in zip(source_files, target_files):
        reduce_dimensionality(pca, source_path, target_path, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True, type=str,
                        help='Please specify pretrained language model path')
    parser.add_argument('-t', '--model_type', required=False, default='MiniLM-L12-v2',
                        help='Please specify model type')
    parser.add_argument('-ml', '--max_length', required=False, type=int, default=256,
                        help='Please specify the max length of input')
    parser.add_argument('-i', '--input_path', required=True, type=str,
                        help='Please specify input file path')
    parser.add_argument('-o', '--output_path', required=False, default='',
                        help='Please specify output file path')
    parser.add_argument('-s', '--sep', required=False, default='|',
                        help='Please specify the seperator for input and output')
    parser.add_argument('-lb', '--llm_batch_size', required=False, type=int,
                        default=512, help='batch size for llm encoding')
    parser.add_argument('-pb', '--pca_batch_size', required=False, type=int,
                        default=10000, help='batch size for pca')
    parser.add_argument('-pt', '--pca_target_dim', required=False, type=int,
                        default=64, help='target dimension for pca')
    parser.add_argument('-pfn', '--precision_float_number', required=False, type=int,
                        default=8, help='precision float for vector')
    parser.add_argument('-mode', '--mode', required=False, type=str,
                        default='encode_and_pca', help='target dimension for pca')
    args, unknown = parser.parse_known_args()

    print('unknown arguments: ', unknown)
    train_config_file = "/opt/huawei/schedule-train/algorithm/train.config"
    with open(train_config_file, 'r', encoding='utf-8') as fin:
        train_config = json.load(fin)
    for param, config in train_config.items():
        if param in args.__dict__:
            args.__dict__[param] = config
    print('arguments: ', args)

    if args.mode == "encode_and_pca":
        encode_fun(args)
        pca_fun(args)
    elif args.mode == "encode":
        encode_fun(args)
    elif args.mode == "pca":
        print(f"the encode data must be stored in {args.output_path}/encode ")
        pca_fun(args)
    elif args.mode == "generate":
        generator_fun(args)


if __name__ == '__main__':
    main()
