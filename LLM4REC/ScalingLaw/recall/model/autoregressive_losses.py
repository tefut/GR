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

from collections import OrderedDict
from typing import Optional

import abc
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from recall.model.negative_sampler import NegativesSampler
from recall.model.utils import dense_to_jagged_1d


class AutoregressiveLoss(torch.nn.Module):

    @abc.abstractmethod
    def jagged_forward(
            self,
            output_embeddings: torch.Tensor,
            supervision_ids: torch.Tensor,
            supervision_embeddings: torch.Tensor,
            supervision_weights: torch.Tensor,
            negatives_sampler,
    ) -> torch.Tensor:
        """
        Variant of forward() when the tensors are already in jagged format.

        Args:
            output_embeddings: [N', D] x float, embeddings for the current
                input sequence.
            supervision_ids: [N'] x int64, (positive) supervision ids.
            supervision_embeddings: [N', D] x float.
            supervision_weights: Optional [N'] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            (1), loss for the current engaged sequence.
        """
        pass

    @abc.abstractmethod
    def forward(
            self,
            lengths: torch.Tensor,
            output_embeddings: torch.Tensor,
            supervision_ids: torch.Tensor,
            supervision_embeddings: torch.Tensor,
            supervision_weights: torch.Tensor,
            negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
            lengths: [B] x int32 representing number of non-zero elements per row.
            output_embeddings: [B, N, D] x float, embeddings for the current
                input sequence.
            supervision_ids: [B, N] x int64, (positive) supervision ids.
            supervision_embeddings: [B, N, D] x float.
            supervision_weights: Optional [B, N] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            (1), loss for the current engaged sequence.
        """
        pass

    def interaction(
            self,
            input_embeddings: torch.Tensor,
            target_ids: torch.Tensor,
            target_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch._assert(len(input_embeddings.size()) == 2, "len(input_embeddings.size()) must be 2")
        torch._assert(len(target_ids.size()) == 2, "len(target_ids.size()) must be 2")
        if target_embeddings is None:
            target_embeddings = self.get_item_embeddings(target_ids)
        torch._assert(len(target_embeddings.size()) == 3, "len(target_embeddings.size()) must be 3")

        with torch.autocast(enabled=True, dtype=torch.float32, device_type="cuda"):
            return self._ndp_module(
                input_embeddings=input_embeddings,  # [B, self._input_embedding_dim]
                item_embeddings=target_embeddings,  # [1/B, X, self._item_embedding_dim]
                item_sideinfo=None,  # [1/B, X, self._item_sideinfo_dim]
                item_ids=target_ids,
                precomputed_logits=None,
            )


class BCELoss(AutoregressiveLoss):
    def __init__(
            self,
            temperature: float,
            similarity_module,  #: NDPModule,
    ) -> None:
        super().__init__()
        self._temperature: float = temperature
        self._ndp_module = similarity_module

    def jagged_forward(
            self,
            output_embeddings: torch.Tensor,
            supervision_ids: torch.Tensor,
            supervision_embeddings: torch.Tensor,
            supervision_weights: torch.Tensor,
            negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        sampled_ids, sampled_negative_embeddings = negatives_sampler(
            positive_ids=supervision_ids,
            num_to_sample=1,
        )
        positive_logits = self.interaction(
            input_embeddings=output_embeddings,  # [B, D] = [N', D]
            target_ids=supervision_ids.unsqueeze(1),  # [N', 1]
            target_embeddings=supervision_embeddings.unsqueeze(1),  # [N', D] -> [N', 1, D]
        )[0] / self._temperature  # [N']

        sampled_negatives_logits = self.interaction(
            input_embeddings=output_embeddings,  # [N', D]
            target_ids=sampled_ids,  # [N', 1]
            target_embeddings=sampled_negative_embeddings,  # [N', 1, D]
        )[0] / self._temperature  # [N']
        sampled_negatives_valid_mask = (
                supervision_ids != sampled_ids.squeeze(1)
        ).float()  # [N']
        loss_weights = supervision_weights * sampled_negatives_valid_mask
        weighted_losses = (
                                  F.binary_cross_entropy_with_logits(
                                      input=positive_logits,
                                      target=torch.ones_like(positive_logits),
                                      reduction='none',
                                  )
                                  + F.binary_cross_entropy_with_logits(
                              input=sampled_negatives_logits,
                              target=torch.zeros_like(sampled_negatives_logits),
                              reduction='none',
                          )
                          ) * loss_weights * 0.5
        return weighted_losses.sum() / loss_weights.sum()

    def forward(
            self,
            lengths: torch.Tensor,
            output_embeddings: torch.Tensor,
            supervision_ids: torch.Tensor,
            supervision_embeddings: torch.Tensor,
            supervision_weights: torch.Tensor,
            negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
        lengths: [B] x int32 representing number of non-zero elements per row.
        output_embeddings: [B, N, D] x float, embeddings for the current input sequence.
        supervision_ids: [B, N] x int64, (positive) supervision ids.
        supervision_embeddings: [B, N, D] x float.
        supervision_weights: Optional [B, N] x float. Optional weights for
            masking out invalid positions, or reweighting supervision labels.
        negatives_sampler: sampler used to obtain negative examples paired with
            positives.
        Returns:
        (1), loss for the current engaged sequence.
        """
        lengths = lengths.squeeze().int()
        jagged_id_offsets = torch.cat(
            (torch.tensor([0], dtype=lengths.dtype).to(lengths.device), torch.cumsum(lengths, dim=0)), dim=0)

        jagged_supervision_ids = dense_to_jagged_1d(
            supervision_ids.unsqueeze(-1).float(),
            jagged_id_offsets,
        ).squeeze(1)
        jagged_supervision_weights = dense_to_jagged_1d(supervision_weights.unsqueeze(-1),
                                                        jagged_id_offsets).squeeze(1)

        jagged_output_embeddings = dense_to_jagged_1d(output_embeddings, jagged_id_offsets)
        jagged_supervision_embeddings = dense_to_jagged_1d(supervision_embeddings, jagged_id_offsets)

        return self.jagged_forward(
            output_embeddings=jagged_output_embeddings,
            supervision_ids=jagged_supervision_ids,
            supervision_embeddings=jagged_supervision_embeddings,
            supervision_weights=jagged_supervision_weights,
            negatives_sampler=negatives_sampler,
        )


class SampledSoftmaxLoss(AutoregressiveLoss):

    def __init__(
            self,
            num_to_sample: int,
            softmax_temperature: float,
            similarity_module,
            activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self._num_to_sample: int = num_to_sample
        self._softmax_temperature: float = softmax_temperature
        self._ndp_module = similarity_module
        self._activation_checkpoint: bool = activation_checkpoint

    def jagged_forward(
            self,
            output_embeddings: torch.Tensor,
            supervision_ids: torch.Tensor,
            supervision_embeddings: torch.Tensor,
            supervision_weights: torch.Tensor,
            negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:

        sampled_ids, sampled_negative_embeddings = negatives_sampler(
            positive_ids=supervision_ids,
            num_to_sample=self._num_to_sample,
        )

        positive_embeddings = supervision_embeddings

        positive_logits = self.interaction(
            input_embeddings=output_embeddings,  # [B, D] = [N', D]
            target_ids=supervision_ids.unsqueeze(1),  # [N', 1]
            target_embeddings=positive_embeddings.unsqueeze(1),  # [N', D] -> [N', 1, D]
        ) / self._softmax_temperature  # [0]
        sampled_negatives_logits = self.interaction(
            input_embeddings=output_embeddings,  # [N', D]
            target_ids=sampled_ids,  # [N', R]
            target_embeddings=sampled_negative_embeddings,  # [N', R, D]
        )  # [N', R]  # [0]
        sampled_negatives_logits = torch.where(
            supervision_ids.unsqueeze(1) == sampled_ids,  # [N', R]
            -5e4,
            sampled_negatives_logits / self._softmax_temperature,
        )

        jagged_loss = -F.log_softmax(
            torch.cat([positive_logits, sampled_negatives_logits], dim=1), dim=1
        )[:, 0]
        return (jagged_loss * supervision_weights).sum() / supervision_weights.sum()

    def forward(
            self,
            lengths: torch.Tensor,
            output_embeddings: torch.Tensor,
            supervision_ids: torch.Tensor,
            supervision_embeddings: torch.Tensor,
            supervision_weights: torch.Tensor,
            negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
            lengths: [B] x int32 representing number of non-zero elements per row.
            output_embeddings: [B, N, D] x float, embeddings for the current
                input sequence.
            supervision_ids: [B, N] x int64, (positive) supervision ids.
            supervision_embeddings: [B, N, D] x float.
            supervision_weights: Optional [B, N] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            (1), loss for the current engaged sequence.
        """
        lengths = lengths.squeeze().int()
        jagged_id_offsets = torch.cat(
            (torch.tensor([0], dtype=lengths.dtype).to(lengths.device),
             torch.cumsum(lengths, dim=0)), dim=0)

        jagged_supervision_ids = dense_to_jagged_1d(
            supervision_ids.unsqueeze(-1).float(),
            jagged_id_offsets,
        ).squeeze(1)

        args = OrderedDict(
            [
                (
                    "output_embeddings",
                    dense_to_jagged_1d(output_embeddings, jagged_id_offsets)
                ),
                (
                    "supervision_ids",
                    jagged_supervision_ids
                ),
                (
                    "supervision_embeddings",
                    dense_to_jagged_1d(supervision_embeddings, jagged_id_offsets)
                ),
                (
                    "supervision_weights",
                    dense_to_jagged_1d(supervision_weights.unsqueeze(-1),
                                       jagged_id_offsets).squeeze(1)
                ),
                (
                    "negatives_sampler",
                    negatives_sampler
                ),
            ]
        )
        if self._activation_checkpoint:
            return checkpoint(
                self.jagged_forward, *args.values(), use_reentrant=False,
            )
        else:
            return self.jagged_forward(
                output_embeddings=dense_to_jagged_1d(output_embeddings, jagged_id_offsets),
                supervision_ids=jagged_supervision_ids,
                supervision_embeddings=dense_to_jagged_1d(supervision_embeddings, jagged_id_offsets),
                supervision_weights=dense_to_jagged_1d(supervision_weights.unsqueeze(-1),
                                                       jagged_id_offsets).squeeze(1),
                negatives_sampler=negatives_sampler,
            )
