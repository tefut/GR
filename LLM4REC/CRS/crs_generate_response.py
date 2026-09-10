# Created by zhuhong 00390804 on 2023/8/1
import argparse
import sys
import os
import time
from io import open
from pathlib import Path

import numpy as np
from knowledge_generator import chatGLMKG, LLMKG


def eval_file(in_path: str, out_path: str = "") -> None:
    ask_content = open(in_path, 'r', encoding='utf-8').read()
    data = np.array([ask_content])
    t = time.time()
    if args.encoding:
        res = llm_kg.encode_knowledge(data, 4)
        print('knowledge encoding shape', len(res), len(res[0]))
    else:
        res = llm_kg.generate_knowledge(data, 1)
        print(f'PROMPT: {data[0]}\n\n ANSWER: {res[0]}\n\n')
    total_time = time.time() - t
    print(f'data num:{len(data)}, total time: {total_time}, time per sample: {total_time / len(data)}')

    if out_path:
        with open(out_path, 'w', encoding='utf-8') as output:
            i = 0
            for _ in data:
                if args.encoding:
                    output.write(f'{args.sep.join([data[i][0]] + list(map(str, res[i])))}\r\n')
                else:
                    output.write(f'{args.sep.join([data[i][0], res[i]])}\r\n')
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True, type=str, help='Please specify model path')
    parser.add_argument('-t', '--model_type', required=False, default='llama', help='Please specify model type')
    parser.add_argument('-i', '--input_path', required=True, type=str, help='Please specify input file path')
    parser.add_argument('-o', '--output_path', required=False, default='', help='Please specify output file path')
    parser.add_argument('-s', '--sep', required=False, default='\u0001', help='Please specify column seperator')
    parser.add_argument('-p', '--prompt_template', required=False, default='', help='Please specify instruction prompt')
    parser.add_argument('-e', '--encoding', action='store_true', help='knowledge generation or encoding')
    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)

    llm_kg = chatGLMKG(os.path.join(args.model_path, args.model_type), args.prompt_template)
    print('input path', args.input_path)
    if args.encoding:
        source_name = 'knowledge_20'
        target_name = 'encoding_20'
    else:
        source_name = 'prompt'
        target_name = 'knowledge'
    if os.path.isfile(args.input_path):
        # 单文件推理
        print(f'evaluating file {args.input_path}')
        eval_file(args.input_path, args.output_path)
    else:
        # 多个文件推理
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        for name in sorted(os.listdir(args.input_path)):
            if name.startswith(source_name):
                print(f'evaluating file {os.path.join(args.input_path, name)}')
                eval_file(os.path.join(args.input_path, name),
                          os.path.join(args.output_path, name.replace(source_name, target_name)))
