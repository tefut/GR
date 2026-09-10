import os

import torch

from data.concat_dataset import DatasetMusicV2LONGER


def create_data_loader(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        prefetch_factor: int = 128,
        num_workers: int = os.cpu_count(),
) -> torch.utils.data.DataLoader:
    """
    创建一个数据加载器(DataLoader), 用于批量加载数据集。

    :param dataset: 要加载的数据集。
    :param batch_size: 每个批次的样本数量。
    :param prefetch_factor: 预取因子，用于控制预取的数据量。
    :param num_workers: 加载数据时使用的子进程数量, 默认为CPU核心数。
    :return: 一个配置好的DataLoader对象。
    """

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        ds: DatasetMusicV2LONGER = worker_info.dataset
        ds.files = ds.files[worker_id::worker_info.num_workers]
        ds.init_multi_csv_iterator()

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    return data_loader
