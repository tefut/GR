# encoding=utf8
# Created by zhuhong 00390804 on 2023/09/05
import argparse
import sys
import os
import re
import time
from io import open
from pathlib import Path
import numpy as np
import json
import numpy as np
from knowledge_generator import chatGLMKG, LLMKG

from transformers import AutoModel, AutoTokenizer
import streamlit as st
from streamlit_chat import message


@st.cache_resource
def get_model():
    tokenizer0 = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model0 = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model1 = model0.eval()
    return tokenizer0, model1


def write_history(log_path0, info):
    if os.path.isfile(log_path0):
        with open(log_path0, 'a', encoding='utf-8') as wf:
            wf.write(info+'\n\n\n')


def gen_response(input0, history, max_length0=4096, top_p0=0.6, temperature0=0.95):
    with st.empty():
        for response, history in model.stream_chat(tokenizer, input0, history, max_length=max_length0, top_p=top_p0,
                                                   temperature=temperature0):
            query, response = history[-1]
            st.write(response)
    print("Response: ", response)
    return response, history


def gen_knowledge(prompt, log_path0="", max_length0=4096, top_p0=0.6, temperature0=0.95):
    with st.empty():
        history = []
        for response, history in model.stream_chat(tokenizer, prompt, history, max_length=max_length0, top_p=top_p0,
                                                   temperature=temperature0):
            _, response = history[-1]
    print(f'PROMPT: {prompt}\n\n ANSWER: {response}\n\n')
    write_history(log_path0, 'PROMPT: ' + prompt + '\n\n ANSWER: ' + response + '\n\n')

    return response


def gen_encoding(feature_list, log_path0=""):
    feature_emotion = []
    for feature in feature_list:
        if '情绪：' in feature:
            emotion = feature.strip().split('：')[1]
            feature_emotion.append(emotion)
    prompt = ','.join(feature_emotion)
    print(f'prompt of emotion: {prompt}')
    res = llm_eg.encode_knowledge([prompt], 1)
    return res


def gen_rec_response(conver_user_last, rec_res, log_path0=""):
    prompt_pre = "下面是一个用户跟一个音乐推荐机器人的对话记录：\n"
    prompt_pro = "\n****************\n根据以上对话记录，请输出这个机器人的回话，内容是向用户推荐后面这些歌曲，"
    prompt = prompt_pre + conver_user_last + prompt_pro
    item_part = []
    for item in rec_res:
        content = '《' + item[1] + '》-' + item[0] + '（语种是' + item[2] + '，情绪是' + item[3] + '）'
        item_part.append(content)
    prompt = prompt + ','.join(item_part)
    answer = gen_knowledge(prompt, log_path0)
    return answer


def determine(input0, log_path0=""):
    prompt_pre = "如果我对你说："
    prompt_pro = "\n****************\n请问我有表达出想听什么歌吗？请在如下选项中选择答案，并给出解释。\nA、有\nB、没有"
    prompt = prompt_pre + "“" + input0 + "”" + prompt_pro
    answer = gen_knowledge(prompt, log_path0)

    return answer


def gen_feature(conver_user, log_path0=""):
    conver_user_all = ''.join(conver_user)
    prompt_pre = '下面是一个用户跟一个音乐推荐机器人的对话记录：\n'
    prompt_pro = '\n****************\n你觉得用户想听的歌曲明确具有什么特点，请按如下格式输出三首歌曲的歌手、歌名、语种、情绪。\n歌手：\n歌名：\n语种：\n情绪：'
    prompt = prompt_pre + conver_user_all + prompt_pro
    answer = gen_knowledge(prompt, log_path0)
    return answer


def extract_feature(feature_chat, log_path0=""):
    lines = feature_chat.split('\n')
    feature_list = []
    for line in lines:
        patterns = ['歌手：', '歌名：', '语种：', '情绪：']
        pattern = "|".join(patterns)
        if re.search(pattern, line):
            if len(line) > 4:
                feature_list.append(line)
    return feature_list


def get_candidate(features, item_path0):
    candidates = []
    encoding_matrix = []
    with open(item_path0, 'r', encoding='utf-8') as rf:
        for line in rf:
            data_dict = json.loads(line.strip())
            if data_dict['歌手'] not in features:
                continue
            candidates.append([data_dict['歌手'], data_dict['歌名'], data_dict['语种'], data_dict['情绪']])
            encoding_matrix.append(data_dict['encoding'])
    return candidates, np.array(encoding_matrix)


def gen_rec(feature_list, item_path0, log_path0=""):
    encoding = gen_encoding(feature_list, log_path0)
    encoding = np.transpose(np.array(encoding))
    features = '\n'.join(feature_list)
    candi_info, encoding_matrix = get_candidate(features, item_path0)
    if len(candi_info) == 0:
        print("Do not get any recomendation in candidates!")
        return []

    cos_score = np.dot(encoding_matrix, encoding)/(np.linalg.norm(encoding_matrix, axis=1).reshape(-1, 1)
                                                   * np.linalg.norm(encoding))

    topk = 3
    top_index = cos_score.reshape(-1).argsort()[-topk:][::-1]
    print(f'The recommended 3 songs are: \n')
    rec_res = []
    for i in top_index:
        print(i)
        print(candi_info[i])
        rec_res.append(candi_info[i])
    return rec_res


def predict(input0, max_length0, top_p0, temperature0, history=None):
    if len(history) < 1:
        history = [('请扮演一个音乐推荐机器人', '好的,我将扮演一个音乐推荐机器人。')]

    with container:
        conver_user_all = []
        print("History: ", history)
        if len(history) > 0:
            for i, (query, response) in enumerate(history[1:]):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))
                conver_user_all.append("用户：" + query + '\n')

        message(input0, avatar_style="big-smile", key=str(len(history)) + "_user")
        print("Input: ", input0)
        conver_user_all.append("用户：" + input0)

        st.write("AI正在回复:")

        # 判断是否需要做推荐
        print(f'Determine whether there is enough information for rec.')
        answer = determine(input0, log_path)
        if "A、有" in answer:
            print(f'There is enough information for rec!')
            # 生成用户喜欢的song的特点
            feature_chat = gen_feature(conver_user_all, log_path)
            # 提取正向song的歌手、歌名、语种、情绪
            feature_list = extract_feature(feature_chat, log_path)
            if len(feature_list) == 0:
                print(f'No feature is extracted!!!\n\n')
                response, history = gen_response(input0, history, max_length=max_length0, top_p=top_p0,
                                                 temperature=temperature0)
            else:
                features = '\n'.join(feature_list)
                print(f'Extracted features are:\n {features}\n\n')
                rec_res = gen_rec(feature_list, item_path, log_path)
                if len(rec_res) == 0:
                    response, history = gen_response(input0, history, max_length=max_length0, top_p=top_p0,
                                                     temperature=temperature0)
                else:
                    answer = gen_rec_response(input0, rec_res, log_path)
                    st.write(answer)
                    history.append((input0, answer))
        else:
            response, history = gen_response(input0, history, max_length=max_length0, top_p=top_p0,
                                             temperature=temperature0)

    return history


if __name__ == '__main__':
    model_path = r"/opt/huawei/workspace/zhuhong/llm/chatglm2-6b"
    encode_model = r"/opt/huawei/workspace/zhuhong/llm/all-MiniLM-L6-v2"
    log_path = r"/opt/huawei/workspace/zhuhong/llm/crs/log/log.txt"
    item_path = r"/opt/huawei/workspace/zhuhong/llm/crs/item/song_encoding.txt"

    st.set_page_config(
        page_title="llm对话演示",
        page_icon=":robot:"
    )

    MAX_TURNS = 20
    MAX_BOXES = MAX_TURNS * 2

    container = st.container()

    # create a prompt text for the text generation
    prompt_text = st.text_area(label="用户命令输入",
                           height=100,
                           placeholder="请在这儿输入您的命令")

    max_length = st.sidebar.slider(
        'max_length', 0, 4096, 2048, step=1
    )
    top_p = st.sidebar.slider(
        'top_p', 0.0, 1.0, 0.6, step=0.01
    )
    temperature = st.sidebar.slider(
        'temperature', 0.0, 1.0, 0.95, step=0.01
    )

    if 'state' not in st.session_state:
        st.session_state['state'] = []

    tokenizer, model = get_model()
    llm_eg = LLMKG(encode_model, '')

    if st.button("发送", key="predict"):
        with st.spinner("AI正在思考，请稍等........"):
            # text generation
            st.session_state["state"] = predict(prompt_text, max_length, top_p, temperature, st.session_state["state"])
