import argparse

import pandas as pd


def transfer_data_format(args):
    input_file = args.input_file
    output_file = args.output_file
    sep = args.sep
    top_p = float(args.top_p)
    top_k = int(args.top_k)

    df = pd.read_csv(input_file, sep=sep, names=["item_id", "sim_item", "score"])
    print(df.head(5))

    p_df = df[(df["item_id"] != df["sim_item"]) & (df["score"] > top_p)]

    p_df['value'] = p_df["sim_item"].str.cat(p_df["score"].astype(str), sep=":")
    print(p_df.head(5))

    k_df = p_df.groupby("item_id")['value'].apply(lambda x: "#".join(x[:top_k])).reset_index()
    print(k_df.head(5))

    k_df.to_csv(output_file, sep="|", index=False, header=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True, type=str)
    parser.add_argument('-o', '--output_file', required=True, type=str)
    parser.add_argument('-s', '--sep', required=True, type=str)
    parser.add_argument('-k', '--top_k', required=True, type=int, default=100)
    parser.add_argument('-p', '--top_p', required=True, type=float, default=0.0)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)

    transfer_data_format(args)


if __name__ == '__main__':
    main()
