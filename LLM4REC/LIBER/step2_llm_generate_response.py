import argparse
import os
import time
from io import open
from typing import List

import numpy as np
from knowledge_generator import chatGLMKG


def generate_response(in_path: str, out_path: str = "",
                      file_names: List[str] = None,
                      llm_batch_size: int = 64):
    print("file_name list:", file_names)
    for name in file_names:
        print("file_name:", name)
        source_file = os.path.join(in_path, name)
        lines = open(source_file, 'r', encoding='utf-8', newline='\r\n').readlines()
        data = np.array([line.rstrip('\r\n').split(args.sep) for line in lines])
        print("input shape:", data.shape)
        t = time.time()
        res = llm_kg.generate_knowledge(data[:, -1], llm_batch_size)
        print("generate example")
        for i in range(3):
            print("i:", i)
            print("input:", data[i])
            print("output:", res[i])
        total_time = time.time() - t
        print(f'data num:{len(data)}, total time: {total_time}, time per sample: {total_time / len(data)}')
        print("begin save")
        target_path = os.path.join(out_path, 'prompt_output_' + name)
        with open(target_path, 'w', encoding='utf-8') as file:
            print('generate path saved', target_path)
            for uid, prompt in zip(data[:, 0], res):
                file.write(f'{uid}{args.sep}{[prompt]}\r\n')

    return


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
    parser.add_argument('-s', '--sep', required=False, default='|',
                        help='Please specify the seperator for input and output')
    parser.add_argument('-lb', '--llm_batch_size', required=False, type=int,
                        default=64, help='batch size for llm encoding')
    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    llm_kg = chatGLMKG(os.path.join(args.model_path, args.model_type))

    print('input path', args.input_path)
    # 多个文件推理
    os.makedirs(args.output_path, exist_ok=True)
    target_file_names = sorted(os.listdir(args.input_path))
    print("target_file_names:", target_file_names)
    generate_response(args.input_path, args.output_path, target_file_names, args.llm_batch_size)
