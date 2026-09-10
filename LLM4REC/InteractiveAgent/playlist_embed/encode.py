import argparse
import os
import stat
import numpy as np
from web import WebService


def load_data(data_path, args):
    files = os.listdir(data_path)
    data = []
    ids = []
    for file in files:
        file_name = os.path.join(data_path, file)
        print(file_name)

        with open(file_name, 'r', encoding='utf-8') as fin:
            for line in fin:
                part = line.strip().split(args.sep)
                playlist_id = part[0]
                ids.append(playlist_id)

                if len(part) < 2 or len(part[1].strip()) <= 0:
                    print(f"the row data is error. {line}")
                    continue

                title = part[1].strip()
                text = f"{title}"

                if len(part) == 4:
                    genre_name = part[2].strip()
                    language_name = part[3].strip()

                    text += f";偏好风格：{genre_name}" if len(genre_name) > 0 and genre_name != "null" else ""
                    text += f";偏好语种：{language_name}" if len(language_name) > 0 and language_name != "null" else ""
                data.append(text)
    return ids, data


def write_to_file(save_file, mode='w'):
    flags = os.O_WRONLY | os.O_CREAT
    stats = stat.S_IWUSR | stat.S_IRUSR
    file_hander = os.fdopen(os.open(save_file, flags, stats), mode, encoding='utf-8')
    return file_hander


def l2_norm(lst):
    array = np.array(lst)
    _norm = np.linalg.norm(array)
    norm_array = array / _norm

    return norm_array


def encode(args):
    batch_size = args.batch_size

    # 加载数据
    ids, data = load_data(args.data_path, args)
    print(f"the length of data: {len(data)}")

    # 构建web请求对象
    infer_service = WebService(url=args.url,
                               app_id=args.app_id,
                               flow_id=args.flow_id,
                               sign_key=args.sign_key)

    # 数据批处理
    output = write_to_file(args.output_file)
    index = 0
    while (index < len(ids)):
        batch_text = data[index:index + batch_size]
        batch_ids = ids[index:index + batch_size]

        responses = infer_service.multi_thread_pangu_api(batch_text, args.max_workers)

        for response in responses:
            if response is None:
                continue

            _embed = response.get("embedding", None)
            _text = response.get("text", None)

            if _embed is None:
                continue

            _embed = l2_norm(_embed)

            id_index = batch_text.index(_text)
            _ids = batch_ids[id_index]
            embed_str = ",".join(list(map(str, _embed)))
            _text = _text.replace("|", ",")  # 数据中保护分隔符，先进行替换
            output.write(f"{_ids}|{_text}|{embed_str}\n")

        index += batch_size
    print("success!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', required=True, type=str)
    parser.add_argument('-a', '--app_id', required=True, type=str)
    parser.add_argument('-s', '--sign_key', required=True, type=str)
    parser.add_argument('-f', '--flow_id', required=True, type=str)
    parser.add_argument('-b', '--batch_size', required=True, type=int)
    parser.add_argument('-d', '--data_path', required=True, type=str)
    parser.add_argument('-o', '--output_file', required=True, type=str)
    parser.add_argument('-se', '--sep', required=True, type=str)
    parser.add_argument('-mw', '--max_workers', required=True, type=int)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    output_path = os.path.dirname(args.output_file)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Make dir: {output_path}")

    encode(args)


if __name__ == "__main__":
    main()
