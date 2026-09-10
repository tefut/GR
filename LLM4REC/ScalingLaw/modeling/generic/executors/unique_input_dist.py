#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, TypeVar, Optional
import os

import torch
import torch.distributed as dist
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable
from torchrec.distributed.types import Awaitable, LazyAwaitable
from torchrec.distributed.embedding_types import KJTList
try:
    torch.ops.load_library(os.path.join(os.path.dirname(__file__), "libhybrid_cpp.so"))
except Exception as ex:
    logging.error(f"File libhybrid_cpp.so not found {ex}")
from concurrent.futures import ThreadPoolExecutor


class ThreadPoolExecutorSingleton:
    _instance: "ThreadPoolExecutorSingleton" = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ThreadPoolExecutorSingleton, cls).__new__(
                cls, *args, **kwargs
            )
            DEFAULT_POST_INPUT_THREADS = 24
            MAX_THREADS = int(
                os.environ.get("POST_INPUT_THREADS", DEFAULT_POST_INPUT_THREADS)
            )
            cls.executor = ThreadPoolExecutor(MAX_THREADS)
        return cls._instance


def get_feature_len_groupby_table_name(grouped_embedding_configs):
    table_names = []
    features_len_by_table_name = [0]
    for group_config in grouped_embedding_configs:
        for table_config in group_config.embedding_tables:
            table_names.append(table_config.name)
            features_len_by_table_name.append(table_config.num_features())
    features_len_by_table_name = features_len_by_table_name[1:]
    return table_names, features_len_by_table_name


class EmptyKJTAwaitable(LazyAwaitable[KeyedJaggedTensor]):
    def __init__(self, kjt: KeyedJaggedTensor) -> None:
        super().__init__()
        self._kjt = kjt

    def _wait_impl(self) -> KeyedJaggedTensor:
        return self._kjt


class BasePostInputProcess(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> KeyedJaggedTensor:
        return EmptyKJTAwaitable(sparse_features)


class SparseFeaturesPostDist(torch.nn.Module):
    def __init__(
        self, feature_processor: Optional[BasePostInputProcess] = None
    ) -> None:
        super().__init__()
        self._dist = feature_processor

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        return self._dist(sparse_features)


EMPTY_POST_INPUT_DIST = SparseFeaturesPostDist(BasePostInputProcess())


def split_keys_offset(offsets, feature_split_by_table: List[int]):

    result = [0 for _ in range(len(feature_split_by_table) + 1)]
    start = 0
    for ind, feat_num in enumerate(feature_split_by_table):
        end = start + feat_num
        result[ind + 1] = offsets[end]
        start = end
    return torch.LongTensor(result)


def recompute_unique_inverse(offset, unique_offset, unique_invserse):

    segment_lengths = offset[1:] - offset[:-1] 
    segment_idx = torch.arange(len(segment_lengths)) 
    offsets_per_segment = unique_offset[:-1]  
    expanded_idx = torch.repeat_interleave(segment_idx, segment_lengths)
    element_offsets = torch.gather(
        offsets_per_segment, 
        dim=0, 
        index=expanded_idx
    )

    return unique_invserse + element_offsets


def do_unique_out(
    origin_kjt: KeyedJaggedTensor,
    feature_split_by_table: List[int],
    device: Optional[torch.device] = None,
):  
    num_of_table = len(feature_split_by_table)
    ids = origin_kjt.values()

    ids_cpu = ids.cpu()
    ids_cpu = torch.LongTensor(ids_cpu)
    offsets_per_key = origin_kjt.offset_per_key()
    unique = torch.empty_like(ids_cpu, pin_memory=True).cpu()
    unique_inverse = torch.empty_like(ids_cpu, pin_memory=True).cpu()
    unique_offset = torch.zeros(num_of_table + 1).long().cpu()

    offsets_per_key_split = split_keys_offset(offsets_per_key, feature_split_by_table)

    # unique
    for table_i in range(num_of_table):
        ids_mapper = torch.classes.hybrid.IdsMapper(2000000) 
        ids_mapper.ids2indices_unique_out(
            ids_cpu, offsets_per_key_split, unique, unique_inverse, unique_offset, table_i
        )

    unique_offset_list_single = unique_offset.tolist()
    unique_offset_list = []
    for table_i in range(num_of_table):
        unique_offset_list.extend(
            [unique_offset_list_single[table_i]] * feature_split_by_table[table_i]
        )
    unique_offset_list.append(unique_offset_list_single[-1])
    unique_offset = torch.LongTensor(unique_offset_list)
    unique.resize_(unique_offset_list[-1])
    unique_inverse = recompute_unique_inverse(offsets_per_key_split, unique_offset, unique_inverse)
    if unique.shape[0] == 0:
        unique = torch.tensor([0])
    unique_kjt = KeyedJaggedTensor(
        keys=origin_kjt.keys(),
        values=unique.pin_memory(),
        offsets=unique_offset.pin_memory(), # offset per key actually
        lengths=origin_kjt.lengths(),
        stride=origin_kjt.stride(),
        length_per_key=origin_kjt.length_per_key(),
    ).to(device=device, non_blocking=True)
    unique_inverse = unique_inverse.pin_memory().to(device=device, non_blocking=True)

    return unique_kjt, unique_inverse


class UniqueKJTAwaitable(LazyAwaitable):
    def __init__(
        self,
        origin_kjt: KeyedJaggedTensor,
        feature_split_by_table: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.future = ThreadPoolExecutorSingleton().executor.submit(
            do_unique_out, origin_kjt, feature_split_by_table, device
        )

    def _wait_impl(self):
        return self.future.result()


class UniqueFeatureProcess(BasePostInputProcess):
    def __init__(
        self,
        table_names: List[str],
        feature_split_by_table: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.table_names = table_names
        self.feature_split_by_table = feature_split_by_table
        self.device = device

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[KeyedJaggedTensor]:
        
        return UniqueKJTAwaitable(
            sparse_features, self.feature_split_by_table, self.device
        )
