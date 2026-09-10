#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torchrec.distributed.embedding_sharding import BaseSparseFeaturesDist

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.distributed.sharding.rw_sequence_sharding import RwSequenceEmbeddingSharding
from torchrec.distributed.sharding.tw_sequence_sharding import TwSequenceEmbeddingSharding
from modeling.generic.executors.unique_input_dist import (
    SparseFeaturesPostDist,
    UniqueFeatureProcess,
    get_feature_len_groupby_table_name,
)


class UniqueRwSequenceEmbeddingSharding(RwSequenceEmbeddingSharding):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_unique_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:

        table_names, features_split_by_table_name = get_feature_len_groupby_table_name(
            self._grouped_embedding_configs
        )
        feature_processor = UniqueFeatureProcess(
            table_names, features_split_by_table_name, device
        )
        return SparseFeaturesPostDist(feature_processor)
    
    
class UniqueTwSequenceEmbeddingSharding(TwSequenceEmbeddingSharding):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_unique_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:

        table_names, features_split_by_table_name = get_feature_len_groupby_table_name(
            self._grouped_embedding_configs
        )
        feature_processor = UniqueFeatureProcess(
            table_names, features_split_by_table_name, device
        )
        return SparseFeaturesPostDist(feature_processor)
