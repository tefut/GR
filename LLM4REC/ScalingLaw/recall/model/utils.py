# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple
import torch


def handle_padded_qk(B: int, cached_k, cached_q, delta_x_offsets: Tuple[torch.Tensor, torch.Tensor], k, n: int, q):
    """
    处理填充的Q和K, 以适应变长序列的注意力机制。

    在自注意力(Self-Attention)机制中, 为了处理不同长度的序列, 需要对序列进行填充(Padding)。
    此函数用于将缓存中的Q(Query)和K(Key)与实际的Q和K进行对齐, 以便正确计算注意力分数。

    参数:
    - B: 批次大小(Batch size)。
    - cached_k: 缓存中的Key张量。
    - cached_q: 缓存中的Query张量。
    - delta_x_offsets: 用于计算偏移量的张量。
    - k: 实际的Key张量。
    - n: 序列长度。
    - q: 实际的Query张量。

    返回:
    - padded_k: 填充后的Key张量。
    - padded_q: 填充后的Query张量。
    """
    padded_q, padded_k = cached_q, cached_k
    # 将偏移量张量展平，并计算实际的索引位置
    flattened_offsets = delta_x_offsets[1] + torch.arange(start=0, end=B * n, step=n,
                                                          device=delta_x_offsets[1].device,
                                                          dtype=delta_x_offsets[1].dtype)
    # 使用索引复制q到padded_q中，然后恢复其原始形状
    padded_q = padded_q.view(B * n, -1).index_copy_(
        dim=0, index=flattened_offsets, source=q,
    ).view(B, n, -1)
    # 使用索引复制k到padded_k中，然后恢复其原始形状
    padded_k = padded_k.view(B * n, -1).index_copy_(
        dim=0, index=flattened_offsets, source=k,
    ).view(B, n, -1)
    return padded_k, padded_q


def batch_gather_embeddings(
        rowwise_indices: torch.Tensor,
        embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        rowwise_indices: (B, N) x int, where each entry is in [0, X).
        embeddings: (B, X, D,) x float.

    Returns:
        (B, N, D,) x float, embeddings corresponding to rowwise_indices.
    """
    _, N = rowwise_indices.size()
    B, X, D = embeddings.size()
    flattened_indices = (
            rowwise_indices
            + torch.arange(
        start=0, end=B, step=1, dtype=rowwise_indices.dtype, device=rowwise_indices.device
    ).unsqueeze(1).expand(-1, N) * X
    )
    return embeddings.view(-1, D)[flattened_indices, :].reshape(rowwise_indices.size() + (D,))


def batch_scatter_embeddings(
        dst_embeddings: torch.Tensor,
        rowwise_indices: torch.Tensor,
        src_embeddings: torch.Tensor,
) -> None:
    """
    Args:
        dst_embeddings: (B, N, D,) x float.
        rowwise_indices: (B,) x int, where each entry is in [0, N - 1).
        source_embeddings: (B, D,) x float.
    """
    B, N, D = dst_embeddings.size()
    flattened_indices = (
            rowwise_indices
            + torch.arange(
        start=0, end=B * N, step=N, dtype=rowwise_indices.dtype, device=rowwise_indices.device
    )
    )
    dst_embeddings.view(B * N, D)[flattened_indices, :] = src_embeddings


def get_current_embeddings(
        lengths: torch.Tensor,
        encoded_embeddings: torch.Tensor,
        curr: int = 1
) -> torch.Tensor:
    """
    Args:
        lengths: (B,) x int
        seq_embeddings: (B, N, D,) x float

    Returns:
        (B, D,) x float, where [i, :] == encoded_embeddings[i, lengths[i] - 1, :]
    """
    B, N, D = encoded_embeddings.size()
    flattened_offsets = (
            (lengths - curr)
            + torch.arange(
        start=0, end=B, step=1, dtype=lengths.dtype, device=lengths.device
    ) * N
    )
    return encoded_embeddings.reshape(-1, D)[flattened_offsets, :].reshape(B, D)


def jagged_or_dense_repeat_interleave_dim0(x: torch.Tensor, lengths: torch.Tensor, repeats: int) -> torch.Tensor:
    if len(x.size()) == 3:
        return x.repeat_interleave(repeats, dim=0)
    else:
        padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
            values=x,
            offsets=[torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)],
            max_lengths=[lengths.max()],
            padding_value=0.0
        )
        lengths = lengths.repeat_interleave(repeats, dim=0)
        return torch.ops.fbgemm.dense_to_jagged(
            padded_x.repeat_interleave(repeats, dim=0),
            [torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)],
        )[0]


def jagged_or_dense_index_select_dim0(x: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if len(x.size()) == 3:
        return x[indices, :, :]
    else:
        padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
            values=x,
            offsets=[torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)],
            max_lengths=[lengths.max()],
            padding_value=0.0,
        )
        return torch.ops.fbgemm.dense_to_jagged(
            padded_x[indices, :],
            [torch.ops.fbgemm.asynchronous_complete_cumsum(lengths[indices])],
        )[0]


def dense_to_jagged_1d(dense, x_offsets, total_l: int = 0):
    jagged_length = int(x_offsets[-1])

    if total_l > 0:
        if total_l >= jagged_length:
            jagged_length = total_l
        else:
            raise ValueError("total_l %s is less than jagged_length %s" % (total_l, jagged_length))

    num_sequences = x_offsets.shape[0] - 1

    jagged_size = [jagged_length] + list(dense.shape[2:])

    jagged_data = torch.full(jagged_size, 0, dtype=dense.dtype, device=dense.device)

    for i in range(num_sequences):
        start = x_offsets[i]
        end = x_offsets[i + 1]
        seq_length = end - start
        jagged_data[start:end] = dense[i, :seq_length]

    return jagged_data
