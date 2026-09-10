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

import abc
import math
from typing import Dict

import torch


class InputFeaturesPreprocessorModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
            self,
            past_lengths: torch.Tensor,
            past_ids: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pass

    def truncated_normal(self, x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        with torch.no_grad():
            size = x.shape
            tmp = x.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            x.data.mul_(std).add_(mean)
            return x


class LearnablePositionalEmbeddingInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):

    def __init__(
            self,
            max_sequence_len: int,
            embedding_dim: int,
            dropout_rate: float,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len, self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.reset_state()

    def debug_str(self) -> str:
        return f"posi_d{self._dropout_rate}"

    def reset_state(self):
        self.truncated_normal(
            self._pos_emb.weight.data, mean=0.0, std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
            self,
            past_lengths: torch.Tensor,
            past_ids: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = (
                past_embeddings * (self._embedding_dim ** 0.5)
                + self._pos_emb(torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1))
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):

    def __init__(
            self,
            max_sequence_len: int,
            item_embedding_dim: int,
            dropout_rate: float,
            rating_embedding_dim: int,
            num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim + rating_embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len, self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings, rating_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"posir_d{self._dropout_rate}"

    def reset_state(self):
        self.truncated_normal(
            self._pos_emb.weight.data, mean=0.0, std=math.sqrt(1.0 / self._embedding_dim),
        )
        self.truncated_normal(
            self._rating_emb.weight.data, mean=0.0, std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
            self,
            past_lengths: torch.Tensor,
            past_ids: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = (
                torch.cat(
                    [
                        past_embeddings, self._rating_emb(past_payloads["ratings"].int())
                    ], dim=-1,
                ) * (self._embedding_dim ** 0.5)
                + self._pos_emb(torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1))
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class CombinedItemAndRatingInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):

    def __init__(
            self,
            max_sequence_len: int,
            item_embedding_dim: int,
            dropout_rate: float,
            rating_embedding_dim: int,
            num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim
        self._rating_embedding_dim: int = rating_embedding_dim
        # Due to [item_0, rating_0, item_1, rating_1, ...]
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len * 2, self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings, rating_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"combir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        self.truncated_normal(
            self._pos_emb.weight.data, mean=0.0, std=math.sqrt(1.0 / self._embedding_dim),
        )
        self.truncated_normal(
            self._rating_emb.weight.data, mean=0.0, std=math.sqrt(1.0 / self._embedding_dim),
        )

    def get_preprocessed_ids(
            self,
            past_lengths: torch.Tensor,
            past_ids: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x int64.
        """
        B, N = past_ids.size()
        return torch.cat(
            [
                past_ids.unsqueeze(2),  # (B, N, 1)
                past_payloads["ratings"].to(past_ids.dtype).unsqueeze(2)
            ], dim=2,
        ).reshape(B, N * 2)

    def get_preprocessed_masks(
            self,
            past_lengths: torch.Tensor,
            past_ids: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x bool.
        """
        B, N = past_ids.size()
        return (past_ids != 0).unsqueeze(2).expand(-1, -1, 2).reshape(B, N * 2)

    def forward(
            self,
            past_lengths: torch.Tensor,
            past_ids: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = torch.cat(
            [
                past_embeddings,  # (B, N, D)
                self._rating_emb(past_payloads["ratings"].int())
            ], dim=2,
        ) * (self._embedding_dim ** 0.5)
        user_embeddings = user_embeddings.view(B, N * 2, D)
        user_embeddings = (
                user_embeddings
                + self._pos_emb(torch.arange(N * 2, device=past_ids.device).unsqueeze(0).repeat(B, 1))
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = self.get_preprocessed_masks(
            past_lengths, past_ids, past_embeddings, past_payloads,
        ).unsqueeze(2).float()  # (B, N * 2, 1,)
        user_embeddings *= valid_mask
        return past_lengths * 2, user_embeddings, valid_mask


class HybridInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):

    def __init__(
            self,
            max_sequence_len: int,
            embedding_dim: int,
            dropout_rate: float,
            num_item_features_ids: int = 0,
            num_item_features: int = 0
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len, self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._item_features_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_item_features_ids, self._embedding_dim,
            padding_idx=0,
        )
        self._num_item_features = num_item_features
        self.reset_state()

    def debug_str(self) -> str:
        return f"hybrid_f{self._num_item_features}"

    def combine_item_info(self,
                          past_lengths: torch.Tensor,
                          past_ids: torch.Tensor,
                          past_embeddings: torch.Tensor,
                          past_payloads: Dict[str, torch.Tensor],
                          ) -> torch.Tensor:
        B, N = past_ids.size()
        item_features_embedding = self._item_features_emb(
            past_payloads["item_features"].int()).view(B, N, self._num_item_features, -1)
        item_features_embedding = torch.mean(item_features_embedding, dim=2)
        output = torch.mean(torch.stack([past_embeddings, item_features_embedding]), dim=0)
        return output

    def reset_state(self):
        self.truncated_normal(
            self._pos_emb.weight.data, mean=0.0, std=math.sqrt(1.0 / self._embedding_dim),
        )
        self.truncated_normal(
            self._item_features_emb.weight.data, mean=0.0, std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
            self,
            past_lengths: torch.Tensor,
            past_ids: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)
        past_embeddings = self.combine_item_info(past_lengths, past_ids, past_embeddings, past_payloads)
        user_embeddings = (
                past_embeddings * (self._embedding_dim ** 0.5)
                + self._pos_emb(torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1))
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class BehaviorInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):

    def __init__(
            self,
            max_sequence_len: int,
            embedding_dim: int,
            dropout_rate: float,
            num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len, self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._num_ratings = num_ratings
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings, self._embedding_dim,
            padding_idx=0,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"behavior_f{self._num_ratings}"

    def combine_item_info(self,
                          past_lengths: torch.Tensor,
                          past_ids: torch.Tensor,
                          past_embeddings: torch.Tensor,
                          past_payloads: Dict[str, torch.Tensor],
                          ) -> torch.Tensor:
        B, N = past_ids.size()
        ratings_embedding = self._rating_emb(past_payloads["ratings"].int())
        output = torch.mean(torch.stack([past_embeddings, ratings_embedding]), dim=0)
        return output

    def reset_state(self):
        self.truncated_normal(
            self._pos_emb.weight.data, mean=0.0, std=math.sqrt(1.0 / self._embedding_dim),
        )
        self.truncated_normal(
            self._rating_emb.weight.data, mean=0.0, std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
            self,
            past_lengths: torch.Tensor,
            past_ids: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)
        past_embeddings = self.combine_item_info(past_lengths, past_ids, past_embeddings, past_payloads)
        user_embeddings = (
                past_embeddings * (self._embedding_dim ** 0.5)
                + self._pos_emb(torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1))
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask
