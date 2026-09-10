import os
import stat
import argparse


def write_to_file(save_file, mode='w'):
    flags = os.O_WRONLY | os.O_CREAT
    stats = stat.S_IWUSR | stat.S_IRUSR
    file_hander = os.fdopen(os.open(save_file, flags, stats), mode)
    return file_hander


def split_domain_items(item_embedding_file):
    data_path = os.path.dirname(item_embedding_file)
    domains_data = {}
    with open(item_embedding_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            name, embed = line.strip().split("|")
            parts = name.split("_")
            domain = parts[0]
            itemid = "_".join(parts[1:])
            if domain not in domains_data:
                domains_data[domain] = {}
            domains_data[domain][itemid] = embed

    for domain, data in domains_data.items():
        domain_item_embedding_file = os.path.join(data_path, f"{domain}.csv")
        item_embeddings_output = write_to_file(domain_item_embedding_file, 'w')
        print(f"{domain}:{len(data)}")

        for item_id, embed in data.items():
            item_embeddings_output.write(item_id + "|" + embed + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--item_embedding_file', required=True, type=str)

    args, unknown = parser.parse_known_args()
    print('unknown arguments: ', unknown)
    print('arguments: ', args)
    item_embedding_file = args.item_embedding_file
    split_domain_items(item_embedding_file)


if __name__ == "__main__":
    main()
