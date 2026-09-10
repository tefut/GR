import os

import torch
from torch.utils.data import IterableDataset
from data.concat_dataset_v1 import DatasetMusicLonger, DatasetAG


def _pad_1d(tensors, target_len, pad_value=0, padding_side="right"):
    """Pad a list of 1D tensors to target_len, then stack.
    padding_side="left": left/leading pad, data right-aligned [0,0,...,actual_data...].
    padding_side="right": right/trailing pad, data left-aligned [actual_data..., 0,0,...].
    Uses F.pad (C++ kernel, thread-parallel) + torch.stack for multi-worker throughput."""
    result = []
    for t in tensors:
        pad_size = target_len - t.shape[0]
        if pad_size > 0:
            if padding_side == "left":
                t = torch.nn.functional.pad(t, (pad_size, 0), value=pad_value)
            else:
                t = torch.nn.functional.pad(t, (0, pad_size), value=pad_value)
        result.append(t)
    return torch.stack(result)


def _pad_2d(tensors, target_outer, pad_value=0, padding_side="right"):
    """Pad a list of 2D tensors on the outer dimension (dim=0) to target_outer.
    Each tensor is [actual_len, inner_dim].
    padding_side="left": pre-padding adds [0, inner_dim] rows at top.
    padding_side="right": post-padding adds [0, inner_dim] rows at bottom.
    Uses F.pad (C++ kernel, thread-parallel) + torch.stack for multi-worker throughput."""
    result = []
    for t in tensors:
        pad_rows = target_outer - t.shape[0]
        if pad_rows > 0:
            if padding_side == "left":
                t = torch.nn.functional.pad(t, (0, 0, pad_rows, 0), value=pad_value)
            else:
                t = torch.nn.functional.pad(t, (0, 0, 0, pad_rows), value=pad_value)
        result.append(t)
    return torch.stack(result)


# 默认非均匀 bucket 表：短序列用细粒度 bucket，长序列用粗粒度 bucket
# 限制总 bucket 数 ≈ 15，确保 NPU kernel 缓存命中率
# 可通过 config data_loader_conf.padding_bucket_table 覆盖
_DEFAULT_BUCKET_TABLE = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def _round_to_bucket(value: int, bucket_size: int, bucket_table: list = None) -> int:
    """Round value up to the nearest bucket boundary.
    
    When bucket_size > 0: round up to nearest multiple of bucket_size (uniform buckets).
    When bucket_size == -1: use non-uniform bucket_table (recommended for NPU).
    When bucket_size == 0: disabled (return value unchanged, pure batch-max).
    
    Non-uniform table provides fine-grained buckets for short sequences (candidates: 1-8)
    and coarse-grained for long sequences (history: 20-150), keeping total unique shapes
    small while minimizing wasted padding.
    """
    if bucket_size == 0 or value <= 0:
        return value
    if bucket_size == -1:
        table = bucket_table or _DEFAULT_BUCKET_TABLE
        for b in table:
            if b >= value:
                return b
        return table[-1]  # fallback
    # Uniform: round up to nearest multiple of bucket_size
    return ((value + bucket_size - 1) // bucket_size) * bucket_size


def _make_ag_dynamic_collate(dataset: DatasetAG, padding_bucket_size: int = 0,
                              padding_bucket_table: list = None,
                              padding_side: str = "right"):
    """Create a collate_fn for DatasetAG with dynamic (batch-longest) padding.
    Pads each batch to the longest sequence in that batch (pure batch-max).
    
    诊断信息通过 _dp_diag key传回主进程（worker子进程中logging/print不可靠）。
    padding_bucket_size: >0 时将 batch-max 向上取整到最近的 bucket_size 倍数，
        减少 NPU kernel 重编译次数。推荐 64。
    """

    # Build field category sets from dataset attributes
    hist_keys = set()
    cand_keys = set()

    # History feature columns
    hist_keys.update(dataset.history_feature_column_names)
    hist_keys.add("history_action_type")
    hist_keys.add("history_ids")
    hist_keys.add(dataset.history_timestamps_column_name)
    hist_keys.add(dataset.history_date_column_name)

    # Candidate feature columns  
    cand_keys.update(dataset.candidate_feature_column_names)
    cand_keys.add("candidate_action_type")
    cand_keys.add("candidate_ids")
    cand_keys.add("loss_weights")
    cand_keys.add("labels")
    cand_keys.add(dataset.candidate_timestamps_column_name)
    cand_keys.add(dataset.candidate_date_column_name)
    cand_keys.add(dataset.candidate_ratings_column_name)

    # Scalar fields (no padding needed)
    scalar_keys = {"uid", "history_lengths", "candidate_lengths"}

    token_per_item = dataset.token_per_item

    def dynamic_collate_fn(batch):
        """Collate variable-length samples, padding to batch-longest."""
        batch_hist_lengths = [s["history_lengths"] for s in batch]
        batch_cand_lengths = [s["candidate_lengths"] for s in batch]

        # Batch-max with optional bucket rounding
        max_hist_len = max(batch_hist_lengths)
        max_cand_len = max(batch_cand_lengths)
        # Round up to nearest bucket boundary to reduce unique shape counts
        max_hist_len = _round_to_bucket(max_hist_len, padding_bucket_size, padding_bucket_table)
        max_cand_len = _round_to_bucket(max_cand_len, padding_bucket_size, padding_bucket_table)
        max_cand_token_len = max_cand_len * token_per_item

        result = {}
        for key in batch[0].keys():
            samples = [s[key] for s in batch]

            # Scalar int (history_lengths, candidate_lengths)
            if key in scalar_keys and not isinstance(samples[0], torch.Tensor):
                result[key] = torch.tensor(samples, dtype=torch.int64)
                continue

            if not isinstance(samples[0], torch.Tensor):
                result[key] = samples  # pass through
                continue

            # 0D scalar tensor (uid)
            if samples[0].dim() == 0:
                result[key] = torch.stack(samples)
                continue

            # 1D sequence tensor
            if samples[0].dim() == 1:
                # Determine target length and pad value
                if key in hist_keys:
                    target_len = max_hist_len
                    pad_val = 0
                elif key in cand_keys:
                    target_len = max_cand_token_len
                    pad_val = 0
                elif key in scalar_keys:
                    # 1D scalar-like (shouldn't happen but handle)
                    result[key] = torch.stack(samples)
                    continue
                else:
                    # Unknown key - try stacking, fallback to pad
                    try:
                        result[key] = torch.stack(samples)
                        continue
                    except RuntimeError:
                        target_len = max(s.shape[0] for s in samples)
                        pad_val = 0

                result[key] = _pad_1d(samples, target_len, pad_value=pad_val, padding_side=padding_side)
                continue

            # 2D tensor (multi-value features: [num_items, max_len_per_item])
            if samples[0].dim() == 2:
                if key in hist_keys:
                    target_outer = max_hist_len
                elif key in cand_keys:
                    target_outer = max_cand_len
                else:
                    target_outer = max(s.shape[0] for s in samples)
                result[key] = _pad_2d(samples, target_outer, padding_side=padding_side)
                continue

            # Fallback: try stack
            result[key] = torch.stack(samples)

        return result

    return dynamic_collate_fn


def create_data_loader(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        prefetch_factor: int = 128,
        num_workers: int = os.cpu_count(),
        use_dynamic_padding: bool = False,
        padding_bucket_size: int = 0,
        padding_bucket_table: list = None,
        padding_side: str = "right",
        pin_memory: bool = False,
        persistent_workers: bool = True,
) -> torch.utils.data.DataLoader:
    """
    创建一个数据加载器(DataLoader), 用于批量加载数据集。

    :param dataset: 要加载的数据集。
    :param batch_size: 每个批次的样本数量。
    :param prefetch_factor: 预取因子，用于控制预取的数据量。
    :param num_workers: 预取时使用的子进程数量, 默认为CPU核心数。
    :param use_dynamic_padding: 是否使用动态padding（按batch最长序列padding）。
    :param padding_bucket_size: 将 batch-max 长度向上取整到最近的 bucket_size 倍数。
        0 表示禁用（纯 batch-max）。推荐 NPU 上设为 64。-1 表示使用非均匀 bucket 表。
    :param padding_bucket_table: 非均匀 bucket 表（当 padding_bucket_size=-1 时生效）。
        None 时使用默认表 [1,2,4,8,16,24,32,48,64,96,128,192,256,384,512]。
    :param padding_side: padding方向，"left"在左侧padding，"right"在右侧padding。默认"right"。
    :param pin_memory: 是否使用页锁定内存。注意：当配合share_memory_()做IPC时，pin会被销毁，
        因此本代码库中pin_memory无H2D收益，默认False。
    :param persistent_workers: 是否保持worker进程跨epoch存活。IterableDataset会自动禁用此选项
        （因为__iter__需要重新初始化文件句柄）。仅对MapDataset有效。
    :return: 一个配置好的DataLoader对象。
    """

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        ds: DatasetMusicLonger = worker_info.dataset
        ds.files = ds.files[worker_id::worker_info.num_workers]
        ds.init_multi_csv_iterator()

    # Dynamic padding: custom collate_fn that pads to batch-longest
    collate_fn = None
    if use_dynamic_padding and isinstance(dataset, DatasetAG):
        collate_fn = _make_ag_dynamic_collate(
            dataset, padding_bucket_size=padding_bucket_size,
            padding_bucket_table=padding_bucket_table,
            padding_side=padding_side)

    # IterableDataset的__iter__每轮需要重新调用init_multi_csv_iterator()重置文件句柄，
    # persistent_workers=True会跳过此重置，导致worker迭代器耗尽后产出空数据。
    # 因此：IterableDataset禁用persistent_workers；MapDataset可安全启用。
    is_iterable = isinstance(dataset, IterableDataset)
    if is_iterable and persistent_workers:
        persistent_workers = False

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 永远不能设置为True！IterableDataset不能shuffle
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    return data_loader
