import torch
from typing import List, Tuple, Dict
from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const


class NegativesSampler(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.embedding_module = None

    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self._maybe_l2_norm(x)

    def load_embedding_module(self, embedding_module: BaseModel):
        self.embedding_module = embedding_module

    def _maybe_l2_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self._l2_norm:
            # x**2 sum in FP16 can overflow; upcast to FP32 for numerical stability
            x_fp32 = x.float()
            squared_sum = torch.sum(x_fp32 ** 2, dim=-1, keepdim=True)
            x = (x_fp32 / torch.clamp(
                torch.sqrt(torch.clamp(squared_sum, 0.0) + 1e-10),
                min=self._l2_norm_eps,
            )).to(x.dtype)
        return x


@ModelRegistry.register(req_hp=True)
class LocalNegativesSampler(NegativesSampler):
    """
    精排模型中用于next item predicition的负采样器
    "hp": {"l2_norm": float,
            "l2_norm_eps": float,
            "num_to_sample": int
            }
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp["feature_conf"]
        self._l2_norm = model_cfg[Const.HP].get("l2_norm")
        self._l2_norm_eps = model_cfg[Const.HP].get("l2_norm_eps")
        self._num_to_sample: int = model_cfg[Const.HP].get("num_to_sample")
        itemid_column = feat_conf["itemid_column"]
        all_item_ids = feat_conf["item_feature_columns"][itemid_column]["feature_count"]
        self._num_items = all_item_ids
        self._feat_conf = feat_conf
        self.register_buffer('_all_item_ids', torch.tensor(range(all_item_ids)))

    def get_all_ids_and_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._cached_ids, self._cached_embeddings

    def debug_str(self) -> str:
        sampling_debug_str = f"local{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        return sampling_debug_str

    def process_batches(self, sampled_ids, model_inputs):
        output_shape = sampled_ids.size()
        device = model_inputs['past_ids'].device
        model_inputs['past_ids'] = sampled_ids.to(device)
        item_feat_conf = self._feat_conf.get("item_feature_columns")
        item_feat_keys = item_feat_conf.keys()
        for item_feat in item_feat_keys:
            feat_enabled = item_feat_conf[item_feat]["enabled"]
            if feat_enabled:
                feat_count = item_feat_conf[item_feat]["feature_count"]
                feat_dtype = item_feat_conf[item_feat]["dtype"]
                if feat_dtype == "con":
                    dtype = torch.float32
                else:
                    dtype = torch.int64
                if feat_dtype == "int":
                    sampled_ids = torch.randint(
                        low=1, high=feat_count,
                        size=output_shape,
                        dtype=dtype,
                        device=device,
                    )
                else:
                    sampled_ids = torch.rand(
                        size=output_shape,
                        dtype=dtype,
                        device=device,
                    )
                model_inputs[item_feat] = sampled_ids
        return model_inputs

    def forward(
            self,
            model_inputs,
            positive_ids
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings).
        """
        output_shape = positive_ids.size() + (self._num_to_sample,)
        sampled_offsets = torch.randint(
            low=1, high=self._num_items,
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        sampled_ids = self._all_item_ids[sampled_offsets.view(-1)].reshape(output_shape)
        sampled_input = self.process_batches(sampled_ids, model_inputs.copy())
        return sampled_ids, self.embedding_module.get_item_embeddings(sampled_input)