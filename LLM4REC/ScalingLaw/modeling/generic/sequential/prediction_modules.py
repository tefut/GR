import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, List, Tuple
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.initialization import truncated_normal
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
import logging


@ModelRegistry.register(multi_sel_multi_subs=[{"FeedForwardModuleLonger"}])
class FeedForwardModule(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        if Const.SUB_MODELS not in model_cfg:
            logging.error("You must assign at least one sub module for FeedForwardModule in the config.")

        self.prediction_modules = nn.ModuleList([
            self.init_sub_model(sub_key)
            for sub_key in model_cfg[Const.SUB_MODELS].keys()
        ])

    def forward(self, x):
        predictions = dict()
        for ffn in self.prediction_modules:
            pred_name, pred = ffn(x)
            predictions[pred_name] = pred
        return predictions


@ModelRegistry.register()
class FeedForwardModuleLonger(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self.name = "rerank_score"
        self._item_emb_dim = model_conf.get("item_embedding_dim", 128)
        

        input_dim = self._item_emb_dim

        self.feed_forward_1 = torch.nn.Linear(in_features=input_dim, out_features=input_dim * 4)
        self.feed_forward_2 = torch.nn.Linear(in_features=input_dim * 4, out_features=8)
        self.out_layer = torch.nn.Linear(in_features=8, out_features=1)
        self.act = torch.nn.ReLU()

        self.reset_params()

    def reset_params(self):
        for name, params in self.named_parameters():
            if 'feed_forward' in name:
                logging.info("Initialize %s as truncated normal: %s params", name, params.data.size())
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                logging.info("Skipping initializing params %s - not configured", name)

    def forward(self, x: torch.Tensor):
        y = self.act(self.feed_forward_1(x))
        y = self.act(self.feed_forward_2(y))
        y = self.out_layer(y)
        y = torch.sigmoid(y).squeeze(2)
        return self.name, y

    
@ModelRegistry.register()
class FeedForwardModuleLongerMOE(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self.name = "rerank_score"
        self._item_emb_dim = model_conf.get("item_embedding_dim", 128)

        # === MoE config ===
        self.num_experts = model_conf.get("num_experts", 4)  # change as needed
        input_dim = self._item_emb_dim
        hidden_dim = input_dim * 4
        mix_dim = input_dim  # keep same as original

        self.router = torch.nn.Linear(in_features=input_dim, out_features=self.num_experts)

        # Experts: each expert reproduces the two-layer FFN up to 'mix_dim'
        experts = []
        for _ in range(self.num_experts):
            experts.append(torch.nn.Sequential(
                torch.nn.Linear(in_features=input_dim, out_features=hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=hidden_dim, out_features=mix_dim),
                torch.nn.ReLU()
            ))
        self.experts = torch.nn.ModuleList(experts)

        # Keep the same final projection and activation as original
        self.out_layer = torch.nn.Linear(in_features=mix_dim, out_features=1)
        self.act = torch.nn.ReLU()

        self.reset_params()

    def reset_params(self):
        for name, params in self.named_parameters():
            # Initialize expert FFNs (layers still contain 'weight'/'bias' under names with 'experts')
            if 'experts' in name:
                logging.info("Initialize %s as truncated normal: %s params", name, params.data.size())
                truncated_normal(params, mean=0.0, std=0.02)
            # Initialize router as well to stabilize early training
            elif 'router' in name:
                logging.info("Initialize %s as truncated normal: %s params", name, params.data.size())
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                # Match prior behavior (e.g., not initializing out_layer here)
                logging.info("Skipping initializing params %s - not configured", name)

    def forward(self, x: torch.Tensor):
        """
        x: shape (B, L, D) where D == item_embedding_dim
        returns: (name, y) where y has shape (B, L), same as original
        """
        B, L, D = x.shape

        # Dense gating over all experts (no top-k)
        # gate: (B, L, E), softmax over experts
        gate_logits = self.router(x)
        gate = torch.softmax(gate_logits, dim=-1)

        # Expert outputs up to mix_dim
        # Collect per-expert hidden outputs: list of (B, L, mix_dim)
        expert_hidden = []
        for expert in self.experts:
            h = expert(x)  # (B, L, mix_dim)
            expert_hidden.append(h)
        # Stack to (B, L, E, mix_dim)
        expert_hidden = torch.stack(expert_hidden, dim=2)

        # Mixture: weighted sum across experts -> (B, L, mix_dim)
        # einsum: gate[b,l,e] * expert_hidden[b,l,e,h] -> y_mix[b,l,h]
        y_mix = torch.einsum('ble,bleh->blh', gate, expert_hidden)

        # Final projection to 1 (same as original), then sigmoid and squeeze
        y = self.out_layer(y_mix)             # (B, L, 1)
        y = torch.sigmoid(y).squeeze(2)       # (B, L)

        return self.name, y
