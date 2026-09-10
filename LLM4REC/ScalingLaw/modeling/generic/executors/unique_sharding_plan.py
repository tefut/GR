#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, List
from torch import nn
from torchrec.distributed.types import (
    ModuleSharder,
)
from modeling.generic.executors.unique_embedding import UniqueEmbeddingCollectionSharder


def get_default_unique_sharders() -> List[ModuleSharder[nn.Module]]:
    return [
        cast(ModuleSharder[nn.Module], UniqueEmbeddingCollectionSharder()),
    ]
