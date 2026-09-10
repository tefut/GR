# encoding=utf8
# Created by zhuhong 00390804 on 2023/8/17
import argparse
import sys
import json
import os
import time
from io import open
from pathlib import Path

import numpy as np
from knowledge_generator import chatGLMKG, LLMKG


def save_encoding(item_path, encodings, output_path):
    with open(output_path, 'w', encoding='utf-8') as wf:
        with open(item_path, 'r', encoding='utf-8') as rf:
            index = 0
            for line in rf:
                data_dict = json.loads(line.strip())
                out_dict = {}
                out_dict['歌手'] = data_dict['artist_name_set']
                out_dict['歌名'] = data_dict['song_name']
                out_dict['语种'] = data_dict['langua_tag_name']
                out_dict['情绪'] = data_dict['mood_tag_name']
                out_dict['encoding'] = encodings[index]
                wf.write(json.dumps(out_dict, ensure_ascii=False))
                wf.write('\n')
                
                index += 1
                if index == len(encodings):
                    break


def gen_encoding(prompt, log_path=""):
    t = time.time()
    res = llm_kg.encode_knowledge(prompt, 10)
    print(f'PROMPT: {prompt[0]}\n\n ENCODING: {res[0]}\n\n')
    total_time = time.time() - t
    print(f'data num:{len(prompt)}, total time: {total_time}, time per sample: {total_time / len(prompt)}')

    return res


def gen_prompt(data_dict):
    info = []
    info.append(data_dict['mood_tag_name'])
    return data_dict['mood_tag_name']


def get_item_info(item_path):
    with open(item_path, 'r', encoding='utf-8') as rf:
        index = 0
        prompt = []
        for line in rf:
            data_dict = json.loads(line.strip())
            info = gen_prompt(data_dict)
            print(info)
            prompt.append(info)
            index += 1
    number = len(prompt)
    print(f'The number of item is: {number}.')
    return prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True, type=str, help='Please specify model path')
    parser.add_argument('-t', '--model_type', required=False, default='llama', help='Please specify model type')
    parser.add_argument('-o', '--output_path', required=False, default='', help='Please specify output path')
    parser.add_argument('-s', '--item_path', required=False, default='', help='Please specify item infomation path')
    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)

    if args.model_type in ['chatglm-6b', 'chatglm-v2']:
        llm_kg = chatGLMKG(os.path.join(args.model_path, args.model_type), '')
    else:
        llm_kg = LLMKG(os.path.join(args.model_path, args.model_type), '')
   
    if os.path.isfile(args.item_path):
        # 提取item info, 构造prompt
        prompt0 = get_item_info(args.item_path)
        
        # encoding
        encodings0 = gen_encoding(prompt0)
        
        # save encoding
        save_encoding(args.item_path, encodings0, args.output_path)

