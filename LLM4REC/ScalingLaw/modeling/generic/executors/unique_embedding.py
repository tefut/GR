#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import distributed as dist, nn
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union as TypeUnion,
)
from torchrec.distributed.embedding import (
    ShardedEmbeddingCollection,
    EmbeddingCollectionSharder,
    EmbeddingCollectionAwaitable,
    create_sharding_infos_by_sharding,
    EmbeddingCollectionContext
)
from torchrec.distributed.types import ShardingEnv, LazyAwaitable, Awaitable
from torchrec.modules.embedding_modules import (
    EmbeddingCollection,
)
from torchrec.distributed.types import (
    ParameterSharding,
    ShardingEnv,
)
from torchrec.distributed.sharding.cw_sequence_sharding import (
    CwSequenceEmbeddingSharding,
)
from torchrec.distributed.sharding.dp_sequence_sharding import (
    DpSequenceEmbeddingSharding,
)
from torchrec.distributed.embedding_types import (
    KJTList,
    ShardingType,
)
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingInfo,
)
from torchrec.distributed.types import (
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
    EmbeddingEvent,
)
from torchrec.distributed.utils import maybe_annotate_embedding_event
from torchrec.modules.utils import construct_jagged_tensors
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, JaggedTensor
from modeling.generic.executors.unique_sharding import UniqueRwSequenceEmbeddingSharding
from modeling.generic.executors.unique_sharding import UniqueTwSequenceEmbeddingSharding
from modeling.generic.executors.unique_input_dist import EMPTY_POST_INPUT_DIST


class UniqueEmbeddingCollectionAwaitable(LazyAwaitable[Dict[str, JaggedTensor]]):
    def __init__(
            self,
            awaitables_per_sharding: List[Awaitable[torch.Tensor]],
            features_per_sharding: List[KeyedJaggedTensor],
            embedding_names_per_sharding: List[List[str]],
            ctx: EmbeddingCollectionContext,
            need_indices: bool = False,
            features_to_permute_indices: Optional[Dict[str, List[int]]] = None,
            module_fqn: Optional[str] = None,
            sharding_types: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self._awaitables_per_sharding = awaitables_per_sharding
        self._features_per_sharding = features_per_sharding
        self._need_indices = need_indices
        self._features_to_permute_indices = features_to_permute_indices
        self._embedding_names_per_sharding = embedding_names_per_sharding
        self._ctx = ctx
        self._module_fqn = module_fqn
        self._sharding_types = sharding_types

    def _wait_impl(self) -> Dict[str, JaggedTensor]:
        jt_dict: Dict[str, JaggedTensor] = {}
        for i, (w, f, e) in enumerate(
                zip(
                    self._awaitables_per_sharding,
                    self._features_per_sharding,
                    self._embedding_names_per_sharding,
                )
        ):
            original_features = (
                None
                if i >= len(self._ctx.input_features)
                else self._ctx.input_features[i]
            )
            reverse_indices = (
                None
                if i >= len(self._ctx.reverse_indices)
                else self._ctx.reverse_indices[i]
            )
            seq_vbe_ctx = (
                None if i >= len(self._ctx.seq_vbe_ctx) else self._ctx.seq_vbe_ctx[i]
            )

            with maybe_annotate_embedding_event(
                    EmbeddingEvent.OUTPUT_DIST_WAIT,
                    self._module_fqn,
                    self._sharding_types[i] if self._sharding_types else None,
            ):
                embeddings = w.wait()

            jt_dict.update(
                construct_jagged_tensors(
                    embeddings=embeddings,
                    features=f,
                    embedding_names=e,
                    need_indices=self._need_indices,
                    features_to_permute_indices=self._features_to_permute_indices,
                    original_features=original_features,
                    reverse_indices=reverse_indices,
                    seq_vbe_ctx=seq_vbe_ctx,
                )
            )
        return jt_dict


def create_unique_embedding_sharding(
        sharding_type: str,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
) -> EmbeddingSharding[
    SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
]:
    if sharding_type == ShardingType.TABLE_WISE.value:
        return UniqueTwSequenceEmbeddingSharding(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )
    elif sharding_type == ShardingType.ROW_WISE.value:
        return UniqueRwSequenceEmbeddingSharding(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )
    elif sharding_type == ShardingType.DATA_PARALLEL.value:
        return DpSequenceEmbeddingSharding(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
        )
    elif sharding_type == ShardingType.COLUMN_WISE.value:
        return CwSequenceEmbeddingSharding(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )
    else:
        raise ValueError(f"Sharding not supported {sharding_type}")


class UniqueShardedEmbeddingCollection(ShardedEmbeddingCollection):
    def __init__(self,
                 module: EmbeddingCollection,
                 table_name_to_parameter_sharding: Dict[str, ParameterSharding],
                 env: ShardingEnv,
                 fused_params: Optional[Dict[str, Any]] = None,
                 device: Optional[torch.device] = None,
                 qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
                 use_index_dedup: bool = False,
                 module_fqn: Optional[str] = None,
                 ) -> None:
        super().__init__(module=module,
                         table_name_to_parameter_sharding=table_name_to_parameter_sharding,
                         env=env,
                         fused_params=fused_params,
                         device=device,
                         qcomm_codecs_registry=qcomm_codecs_registry,
                         use_index_dedup=use_index_dedup,
                         module_fqn=module_fqn)
        self._unique_input_dists: List[nn.Module] = []
        self._has_uninitialized_unique_input_dist: bool = True
        sharding_type_to_sharding_infos = create_sharding_infos_by_sharding(
            module,
            table_name_to_parameter_sharding,
            fused_params,
        )
        self._sharding_type_to_sharding: Dict[
            str,
            EmbeddingSharding[
                SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
            ],
        ] = {
            sharding_type: create_unique_embedding_sharding(
                sharding_type=sharding_type,
                sharding_infos=embedding_confings,
                env=env,
                device=device,
                qcomm_codecs_registry=self.qcomm_codecs_registry,
            )
            for sharding_type, embedding_confings in sharding_type_to_sharding_infos.items()
        }

    def _create_unique_input_dist(
            self,
    ) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            if hasattr(sharding, "create_unique_input_dist"):
                self._unique_input_dists.append(sharding.create_unique_input_dist(self._device))
            else:
                self._unique_input_dists.append(EMPTY_POST_INPUT_DIST)

    def unique_input_dist(
            self, ctx: EmbeddingCollectionAwaitable, features: KJTList
    ) -> KJTList:

        if self._has_uninitialized_unique_input_dist:
            self._create_unique_input_dist()
            self._has_uninitialized_unique_input_dist = False
        with torch.no_grad():
            features_list = []
            for p_in_dist, shard_features in zip(self._unique_input_dists, features):
                features_unique = p_in_dist(shard_features)
                features_list.append(features_unique)
            return features_list

    def output_dist(
            self, ctx: EmbeddingCollectionContext, output: List[torch.Tensor]
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        awaitables_per_sharding: List[Awaitable[torch.Tensor]] = []
        features_before_all2all_per_sharding: List[KeyedJaggedTensor] = []
        for odist, embeddings, sharding_ctx in zip(
                self._output_dists,
                output,
                ctx.sharding_contexts,
        ):
            awaitables_per_sharding.append(odist(embeddings, sharding_ctx))

            features_before_all2all_per_sharding.append(
                # pyre-fixme[6]: For 1st argument expected `KeyedJaggedTensor` but
                #  got `Optional[KeyedJaggedTensor]`.
                sharding_ctx.features_before_input_dist
            )
        return UniqueEmbeddingCollectionAwaitable(
            awaitables_per_sharding=awaitables_per_sharding,
            features_per_sharding=features_before_all2all_per_sharding,
            embedding_names_per_sharding=self._embedding_names_per_sharding,
            need_indices=self._need_indices,
            features_to_permute_indices=self._features_to_permute_indices,
            ctx=ctx,
        )


class UniqueEmbeddingCollectionSharder(EmbeddingCollectionSharder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def shard(
            self, module: EmbeddingCollection, params: Dict[str, ParameterSharding],
            env: ShardingEnv, device: Optional[torch.device] = None, module_fqn: Optional[str] = None,
    ) -> UniqueShardedEmbeddingCollection:
        return UniqueShardedEmbeddingCollection(
            module,
            params,
            env,
            self.fused_params,
            device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            use_index_dedup=self._use_index_dedup,
            module_fqn=module_fqn,
        )
