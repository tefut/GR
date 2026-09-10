import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
from typing import Dict, List, Tuple


def get_activation(activation, hidden_units=None):
    if isinstance(activation, str):
        if activation.lower() in ["prelu", "dice"]:
            if type(hidden_units) != int:
                raise TypeError(f"hidden_units 必须是 int 类型，当前类型为 {type(hidden_units).__name__}")
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "softmax":
            return nn.Softmax(dim=-1)
        elif activation.lower() == "prelu":
            return nn.PReLU(hidden_units, init=0.1)
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            if len(activation) != len(hidden_units):
                raise ValueError(f"activation 长度 {len(activation)} 与 hidden_units 长度 {len(hidden_units)} 不匹配")
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation


@ModelRegistry.register()
class SIM_Output_Layer(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp.get("model_conf")
        input_dim = 2 * model_conf.get("item_embedding_dim", 128)
        batch_norm = model_cfg[Const.HP].get("batch_norm", True)
        output_dim = 1
        batch_norm = False
        bn_only_once = False
        use_bias = True
        hidden_activations = "ReLU"
        hidden_units = [64]
        output_activation = None
        dropout_rates = 0.0

        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = get_activation(hidden_activations, hidden_units)
        hidden_units = [input_dim] + hidden_units
        if batch_norm:
            dense_layers.append(nn.LayerNorm(hidden_units[idx + 1]))
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm and not bn_only_once:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.mlp = nn.Sequential(*dense_layers)

    def forward(self, inputs):
        return self.mlp(inputs)


@ModelRegistry.register(req_subs={"SIM_Output_Layer"})
class SIM(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp.get("model_conf")
        attention_dim = model_conf.get("item_embedding_dim", 128)
        num_heads = model_cfg[Const.HP].get("num_heads", 8)
        train_conf = common_hp["train_conf"]
        self.topk = train_conf.get("sim_topk", 50)

        embedding_dim = attention_dim

        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads

        if embedding_dim % self.num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads")

        self.head_dim = self.embedding_dim // self.num_heads

        # gsu 部分
        self.gsu_wQ = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.gsu_wK = nn.Linear(self.embedding_dim, self.embedding_dim)

        # esu 部分
        self.esu_wQ = nn.Linear(self.embedding_dim, self.embedding_dim)  # 输出 D
        self.esu_wK = nn.Linear(self.embedding_dim, self.embedding_dim)  # 输出 D
        self.esu_wV = nn.Linear(self.embedding_dim, self.embedding_dim)  # 输出 D

        # 输出投影层
        self.esu_wO = nn.Linear(self.embedding_dim, self.embedding_dim)

        if "SIM_Output_Layer" in model_cfg[Const.SUB_MODELS]:
            self.dnn_aux = self.init_sub_model("SIM_Output_Layer")
        else:
            raise ValueError("A SIM_Output_Layer should be assigned in sub_models")

        if "SIM_Output_Layer" in model_cfg[Const.SUB_MODELS]:
            self.dnn = self.init_sub_model("SIM_Output_Layer")
        else:
            raise ValueError("A SIM_Output_Layer should be assigned in sub_models")
        self.reset_parameters()

    def reset_parameters(self):
        def default_reset_params(m):
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        def custom_reset_params(m):
            if hasattr(m, 'init_weights'):
                m.init_weights()

        self.apply(default_reset_params)
        self.apply(custom_reset_params)

    def _gsu_soft_search(self, user_seq_emb, target_emb, attention_mask, merge_type='standard_attention'):
        is_train_multi_target = attention_mask.shape[-1] != 1
        target_emb_proj = self.gsu_wQ(target_emb)
        user_seq_emb_proj = self.gsu_wK(user_seq_emb)
        qK = torch.matmul(target_emb_proj, user_seq_emb_proj.transpose(1, 2))  # Shape: [B, C, L]
        mask_for_qk = (attention_mask.squeeze(-1) == 0)
        if not is_train_multi_target:
            mask_for_qk = mask_for_qk.unsqueeze(1)
        k = min(self.topk, user_seq_emb.shape[1])
        if k == 0:
            topk_shape = list(qK.shape[:-1]) + [0, self.embedding_dim]
            gsu_out_topk = torch.zeros(topk_shape, device=user_seq_emb.device, dtype=user_seq_emb.dtype)
        else:
            qK_for_topk = qK.masked_fill(mask_for_qk, -1e4)

            _, indices = torch.topk(qK_for_topk, k, dim=-1, largest=True)

            num_candidates = target_emb.shape[1]
            user_seq_emb_expanded = user_seq_emb.unsqueeze(1).expand(-1, num_candidates, -1, -1)
            gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, self.embedding_dim)
            gsu_out_topk = torch.gather(user_seq_emb_expanded, 2, gather_index)  # B, C, K, D

        original_mask_expanded = mask_for_qk.expand(-1, indices.shape[1], -1)  # [B, C, L]

        topk_mask = torch.gather(original_mask_expanded, 2, indices)  # [B, C, K]

        if merge_type == 'standard_attention':
            qK_for_softmax = qK.masked_fill(mask_for_qk, -1e4)
            attention_weights = torch.softmax(qK_for_softmax, dim=-1)
            attention_weights = attention_weights.nan_to_num(nan=0.0)
            gsu_merged = torch.matmul(attention_weights, user_seq_emb)

        elif merge_type == 'sim_original_masked':
            float_mask = (~mask_for_qk).float()
            qK_for_merge = qK * float_mask
            gsu_merged = torch.matmul(qK_for_merge, user_seq_emb)

        elif merge_type == 'sim_original_unmasked':
            gsu_merged = torch.matmul(qK, user_seq_emb)

        else:
            raise ValueError(f"Unknown merge_type: {merge_type}")

        return gsu_out_topk, gsu_merged, topk_mask

    def _esu_manual_mha(self, sequence_emb, target_emb, attention_mask=None):
        # sequence_emb: [B, C, K, D], target_emb: [B, C, D], attention_mask: [B, C, K]
        B, C, K, D = sequence_emb.shape
        H = self.num_heads
        D_h = self.head_dim  # D_h = D / H

        q_proj = self.esu_wQ(target_emb)  # [B, C, D]
        k_proj = self.esu_wK(sequence_emb)  # [B, C, K, D]
        v_proj = self.esu_wV(sequence_emb)  # [B, C, K, D]

        # Query: [B, C, D] -> [B*C, 1, D] -> [B*C, 1, H, D_h] -> [B*C, H, 1, D_h]
        q = q_proj.view(B * C, 1, H, D_h).transpose(1, 2)

        # Key: [B, C, K, D] -> [B*C, K, D] -> [B*C, K, H, D_h] -> [B*C, H, K, D_h]
        k = k_proj.view(B * C, K, H, D_h).transpose(1, 2)

        # Value: [B, C, K, D] -> [B*C, K, D] -> [B*C, K, H, D_h] -> [B*C, H, K, D_h]
        v = v_proj.view(B * C, K, H, D_h).transpose(1, 2)

        if attention_mask is not None:
            # [B, C, K] -> [B*C, K] -> [B*C, 1, 1, K] 广播 [B*C, H, 1, K]
            attn_mask = (attention_mask.view(B * C, K) == 0).unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = None

        # 输入: Q[B*C, H, 1, D_h], K[B*C, H, K, D_h], V[B*C, H, K, D_h]
        # 输出 [B*C, H, 1, D_h]
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        # FP16 range is ±65504; nan_to_num replacement values must be within range
        _fp16_max = torch.finfo(attn_output.dtype).max if attn_output.dtype == torch.float16 else 1e9
        attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=_fp16_max, neginf=-_fp16_max)

        # [B*C, H, 1, D_h] -> [B*C, 1, H, D_h] -> [B*C, 1, D] -> [B, C, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, C, D)

        esu_output = self.esu_wO(attn_output)  # [B, C, D]
        return esu_output

    def forward(self, item_embeddings, attn_mask, candidate_embedding):
        return self._forward_rerank(item_embeddings, attn_mask, candidate_embedding)

    def _forward_rerank(self, item_embeddings, attn_mask, candidate_embedding):
        # 历史序列
        history_emb = item_embeddings
        history_mask = attn_mask

        # Stage 1: GSU
        gsu_topk_emb, gsu_merged_user_interest, topk_mask = self._gsu_soft_search(history_emb, candidate_embedding,
                                                                                  history_mask)

        # 计算辅助损失
        aux_input = torch.cat([gsu_merged_user_interest, candidate_embedding], dim=-1)  # bs,C,D bs,C,D

        del gsu_merged_user_interest  # 后续不使用gsu_merged_user_interest

        aux_input_normalized = F.normalize(aux_input, p=2, dim=-1, eps=1e-3)

        del aux_input

        y_aux = self.dnn_aux(aux_input_normalized)

        del aux_input_normalized
        # Stage 2: ESU
        esu_user_interest = self._esu_manual_mha(gsu_topk_emb, candidate_embedding, topk_mask)

        del gsu_topk_emb
        # 计算主任务预测
        pred_input = torch.cat([esu_user_interest, candidate_embedding], dim=-1)
        pred_input_normalized = F.normalize(pred_input, p=2, dim=-1, eps=1e-3)
        y_pred = self.dnn(pred_input_normalized)

        return {"y_pred": y_pred, "y_aux": y_aux}
