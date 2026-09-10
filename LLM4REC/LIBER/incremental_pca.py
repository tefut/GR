import argparse
import os
import os.path
import pickle
import stat

import numpy as np
from sklearn.decomposition import IncrementalPCA


def write_to_file(save_file, mode='w'):
    flags = os.O_WRONLY | os.O_CREAT
    stats = stat.S_IWUSR | stat.S_IRUSR
    file_hander = os.fdopen(os.open(save_file, flags, stats), mode)
    return file_hander


def batch_read_data(data, user_data=None, batch_size=1024):
    for i in range(0, len(data), batch_size):
        if user_data is not None:
            yield user_data[i:i + batch_size], data[i:i + batch_size]
        else:
            yield None, data[i:i + batch_size]


def load_data(data_path, sep='|'):
    with open(file=data_path, mode='r') as fd:
        data = [[x.split(sep)[0], list(map(float, x.strip().split(sep)[1].split(',')))] for x in fd.readlines()]
    user_data = [line[0] for line in data]
    emb_data = [line[1] for line in data]
    return np.array(user_data), np.array(emb_data)


def get_incremental_pca(input_data_files, batch_size, target_dim, pca_save_file):
    ipca = IncrementalPCA(n_components=target_dim, batch_size=batch_size)
    fit_iter = 0
    remains = []
    for file in input_data_files:
        print(f'evaluating file {file} for fit PCA')
        user_data, emb_data = load_data(file)
        print(f'loading file {file}, user shape {user_data.shape}, embedding shape {emb_data.shape}')
        for _, batch_emb in batch_read_data(emb_data, batch_size=batch_size):
            if batch_emb.shape[0] == batch_size:
                ipca.partial_fit(batch_emb)
                if fit_iter % 100 == 0:
                    print(fit_iter)
                fit_iter += 1
            else:
                remains.extend(batch_emb)
                if len(remains) > batch_size:
                    ipca.partial_fit(remains[:batch_size])
                    if fit_iter % 100 == 0:
                        print(fit_iter)
                    remains = remains[batch_size:]
                    fit_iter += 1

    # 把PCA参数进行保存
    pickle_file = write_to_file(pca_save_file, 'wb')
    pickle.dump(ipca, pickle_file)
    print(f"save pca in {pca_save_file}")
    return ipca


def reduce_dimensionality(pca, data_or_path, batch_size, save_path, sep='|'):
    trans_iter = 0
    if isinstance(data_or_path, str):
        user_data, emb_data = load_data(data_or_path, sep)
        print(f'loading file {data_or_path} for reduce dimensionality, num {len(user_data)}')
    else:
        user_data, emb_data = data_or_path
    dest = write_to_file(save_path)
    total_num = 0
    for batch_user, batch_emb in batch_read_data(emb_data, user_data, batch_size):
        reduced_emb = pca.transform(batch_emb)
        for user, emb in zip(batch_user, reduced_emb):
            embed_txt = ','.join(list(map(str, emb)))
            dest.write(user + sep + embed_txt + '\n')
        total_num += len(batch_user)
        if trans_iter % 100 == 0:
            print(trans_iter)
        trans_iter += 1
    dest.close()
    print(f"Saved in {save_path}, data num {total_num}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True, type=str,
                        help='Please specify input file path')
    parser.add_argument('-o', '--output_path', required=True, type=str,
                        help='Please specify output file path')
    parser.add_argument('-p', '--data_prefix', required=True, type=str,
                        help='data prefix')
    parser.add_argument('-b', '--batch_size', required=False, type=int,
                        default=10000, help='batch size for pca')
    parser.add_argument('-t', '--target_dim', required=False, type=int,
                        default=64, help='target dimension for pca')
    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    source_files, target_files = [], []
    for name in sorted(os.listdir(args.input_path)):
        if name.startswith(args.data_prefix):
            source_files.append(os.path.join(args.input_path, name))
            target_files.append(os.path.join(args.output_path, name))
    pca_save_path = os.path.join(args.output_path, 'pca_' + args.data_prefix + '.pkl')
    pca = get_incremental_pca(source_files, args.batch_size,
                              args.target_dim, pca_save_path)
    for source_path, target_path in zip(source_files, target_files):
        reduce_dimensionality(pca, source_path, args.batch_size, target_path)


if __name__ == '__main__':
    main()
