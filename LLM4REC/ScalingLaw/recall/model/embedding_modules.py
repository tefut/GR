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
import torch
import json


class EmbeddingModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
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


class LocalEmbeddingModule(EmbeddingModule):

    def __init__(self,
                 feature_name: str,
                 feature_map_file: str,
                 item_embedding_dim: int,
                 ):

        super().__init__()
        self.feature_name = feature_name
        self.feature_map_file = feature_map_file
        self.num_items = self.get_num_items()
        print("num_items: ", self.num_items)

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(self.num_items + 1, item_embedding_dim, padding_idx=0)
        self.reset_params()

    def get_num_items(self):
        with open(self.feature_map_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
        feature_vocab_size = data.get("maxIndexMap", {})
        if self.feature_name not in feature_vocab_size:
            raise RuntimeError(f"{self.feature_name} not in feature_map.")
        return feature_vocab_size.get(self.feature_name)

    def get_feature_map(self):
        with open(self.feature_map_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
        feature_map = data.get("sparse", {})
        if self.feature_name not in feature_map:
            raise RuntimeError(f"{self.feature_name} not in feature_map.")
        return feature_map.get(self.feature_name)

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self):
        for name, params in self.named_parameters():
            if '_item_emb' in name:
                print(f"Initialize {name} as truncated normal: {params.data.size()} params")
                self.truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):

    def __init__(
            self,
            num_items: int,
            item_embedding_dim: int,
            item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(num_items + 1, item_embedding_dim, padding_idx=0)
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self):
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(f"Initialize {name} as truncated normal: {params.data.size()} params")
                self.truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
