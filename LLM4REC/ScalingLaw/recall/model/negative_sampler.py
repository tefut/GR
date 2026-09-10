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


from typing import List, Tuple

import abc
import torch


class NegativesSampler(torch.nn.Module):

    def __init__(self, l2_norm: bool, l2_norm_eps: float) -> None:
        super().__init__()

        self._l2_norm: bool = l2_norm
        self._l2_norm_eps: float = l2_norm_eps

    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self._maybe_l2_norm(x)

    def _maybe_l2_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self._l2_norm:
            squared_sum = torch.sum(x ** 2, dim=-1, keepdim=True)
            x = x / torch.clamp(
                torch.sqrt(torch.clamp(squared_sum, 0.0) + 1e-10),
                min=self._l2_norm_eps,
            )
        return x

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def process_batch(
            self,
            ids: torch.Tensor,
            presences: torch.Tensor,
            embeddings: torch.Tensor,
    ) -> None:
        pass

    @abc.abstractmethod
    def forward(
            self,
            positive_ids: torch.Tensor,
            num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings).
        """
        pass


class LocalNegativesSampler(NegativesSampler):

    def __init__(
            self,
            num_items: int,
            item_emb: torch.nn.Embedding,
            all_item_ids: List[int],
            l2_norm: bool,
            l2_norm_eps: float,
    ) -> None:
        super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)

        self._num_items: int = len(all_item_ids)
        self._item_emb: torch.nn.Embedding = item_emb
        self.register_buffer('_all_item_ids', torch.tensor(all_item_ids))

    def debug_str(self) -> str:
        sampling_debug_str = f"local{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        return sampling_debug_str

    def process_batch(
            self,
            ids: torch.Tensor,
            presences: torch.Tensor,
            embeddings: torch.Tensor,
    ) -> None:
        pass

    def forward(
            self,
            positive_ids: torch.Tensor,
            num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings).
        """
        output_shape = positive_ids.size() + (num_to_sample,)
        sampled_offsets = torch.randint(
            low=0, high=self._num_items,
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        ).int()
        sampled_ids = self._all_item_ids[sampled_offsets.view(-1)].reshape(output_shape)
        return sampled_ids, self.normalize_embeddings(self._item_emb(sampled_ids))


class HybridLocalNegativesSampler(NegativesSampler):

    def __init__(
            self,
            num_items: int,
            item_emb: torch.nn.Embedding,
            side_info_emb: torch.nn.Embedding,
            id_side_dict: torch.nn.Embedding,
            all_item_ids: List[int],
            l2_norm: bool,
            l2_norm_eps: float,
    ) -> None:
        super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)

        self._num_items: int = len(all_item_ids)
        self._item_emb: torch.nn.Embedding = item_emb
        self._side_info_emb: torch.nn.Embedding = side_info_emb
        self._id_side_dict: torch.nn.Embedding = id_side_dict
        self.register_buffer('_all_item_ids', torch.tensor(all_item_ids))

    def debug_str(self) -> str:
        sampling_debug_str = f"hybrid_local{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        return sampling_debug_str

    def process_batch(
            self,
            ids: torch.Tensor,
            presences: torch.Tensor,
            embeddings: torch.Tensor,
    ) -> None:
        pass

    def id_with_side_emb(self, ids: torch.Tensor) -> torch.Tensor:
        id_emb = self._item_emb(ids)
        side_feature = self._id_side_dict(ids)
        side_feature_emb = self._side_info_emb(side_feature)

        side_feature_emb = torch.mean(side_feature_emb, dim=-2)
        output = self.normalize_embeddings(torch.mean(torch.stack([id_emb, side_feature_emb]), dim=0))

        return output

    def forward(
            self,
            positive_ids: torch.Tensor,
            num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings).
        """
        output_shape = positive_ids.size() + (num_to_sample,)
        sampled_offsets = torch.randint(
            low=0, high=self._num_items,
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        ).int()
        sampled_ids = self._all_item_ids[sampled_offsets.view(-1)].reshape(output_shape)
        return sampled_ids, self.id_with_side_emb(sampled_ids)


class InBatchNegativesSampler(NegativesSampler):

    def __init__(
            self,
            l2_norm: bool,
            l2_norm_eps: float,
            dedup_embeddings: bool,
    ) -> None:
        super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)

        self._dedup_embeddings: bool = dedup_embeddings
        self._cached_embeddings = None
        self._cached_ids = None

    def debug_str(self) -> str:
        sampling_debug_str = f"in-batch{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        if self._dedup_embeddings:
            sampling_debug_str += "-dedup"
        return sampling_debug_str

    def process_batch(
            self,
            ids: torch.Tensor,
            presences: torch.Tensor,
            embeddings: torch.Tensor,
    ) -> None:
        """
        Args:
           ids: (N') or (B, N) x int64
           presences: (N') or (B, N) x bool
           embeddings: (N', D) or (B, N, D) x float
        """

        if self._dedup_embeddings:
            valid_ids = ids[presences]
            unique_ids, unique_ids_inverse_indices = torch.unique(input=valid_ids, sorted=False, return_inverse=True)
            device = unique_ids.device
            unique_embedding_offsets = torch.empty(
                (unique_ids.numel(),), dtype=torch.int64, device=device,
            )
            unique_embedding_offsets[unique_ids_inverse_indices] = (
                torch.arange(valid_ids.numel(), dtype=torch.int64, device=device)
            )
            unique_embeddings = embeddings[presences][unique_embedding_offsets, :]
            self._cached_embeddings = self._maybe_l2_norm(unique_embeddings)
            self._cached_ids = unique_ids
        else:
            self._cached_embeddings = self._maybe_l2_norm(embeddings[presences])
            self._cached_ids = ids[presences]

    def get_all_ids_and_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._cached_ids, self._cached_embeddings

    def forward(
            self,
            positive_ids: torch.Tensor,
            num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings,).
        """
        X = self._cached_ids.size(0)
        sampled_offsets = torch.randint(
            low=0, high=X,
            size=positive_ids.size() + (num_to_sample,),
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        ).int()
        return (
            self._cached_ids[sampled_offsets],
            self._cached_embeddings[sampled_offsets]
        )
