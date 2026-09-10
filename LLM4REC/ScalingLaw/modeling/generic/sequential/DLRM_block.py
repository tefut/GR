import abc
import math
from typing import Dict, Tuple, List, Optional, Union, Optional
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.utils.constants import Const
from modeling.model_registry import ModelRegistry


@ModelRegistry.register()
class DNN(BaseModel):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. 

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. 

      Arguments
        - **inputs_dim**: input feature dimension.

        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.dropout_rate = model_cfg[Const.HP].get("dropout_rate", 0)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.seed = model_cfg[Const.HP].get("seed", 1024)
        self.l2_reg = model_cfg[Const.HP].get("l2_reg", 0)
        inputs_dim = model_cfg[Const.HP].get("inputs_dim")
        he_uniform_initial = model_cfg[Const.HP].get("he_uniform_initial", True)
        init_std = model_cfg[Const.HP].get("init_std", 0.0001)
        hidden_units = model_cfg[Const.HP].get("hidden_units")
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = torch.nn.ModuleList(
            [torch.nn.ReLU() for i in range(len(hidden_units))])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                if he_uniform_initial:
                    torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu')
                else:
                    torch.nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        deep_input = inputs
        len_linear = len(self.linears)
        for i in range(len_linear):
            fc = self.linears[i](deep_input)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


@ModelRegistry.register()
class CrossNet(BaseModel):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.

    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.layer_num = model_cfg[Const.HP].get("layer_num", 2)
        seed = model_cfg[Const.HP].get("seed", 1024)
        self.parameterization = model_cfg[Const.HP].get("parameterization", 'vector')
        in_features = model_cfg[Const.HP].get("in_features")
        if self.parameterization == 'vector':
            # weight in DCN.  (in_features, 1)
            self.kernels = torch.nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == 'matrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = torch.nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = torch.nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            torch.nn.init.kaiming_uniform_(self.kernels[i], a=math.sqrt(5))

        for i in range(self.kernels.shape[0]):
            torch.nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            torch.nn.init.zeros_(self.bias[i])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0])).contiguous()
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


@ModelRegistry.register(opt_subs={"SRN"})
class SRN(BaseModel):
    """
    Soft Retargeting Network (SRN) - 相似分桶模块
    PyTorch 实现版本

    参数:
        num_bins: int, 分桶数量
        embedding_size: int, 每个分桶的向量维度
        sim_gate_w: float, 相似门函数的w初始值
        sim_gate_b: float, 相似门函数的b初始值
        sim_gate_trainable: bool, 是否可训练
        gru_units: int or None, 是否使用GRU以及其单元数
        binning_type: str, 分桶方式 'default' 或 'normalize'
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        num_bins = model_cfg[Const.HP].get("num_bins", 10)
        embedding_size = model_cfg[Const.HP].get("embedding_size", 16)
        gru_units = model_cfg[Const.HP].get("gru_units", None)
        sim_gate_w = model_cfg[Const.HP].get("sim_gate_w", 10.0)
        sim_gate_b = model_cfg[Const.HP].get("sim_gate_b", 9.0)
        sim_gate_trainable = model_cfg[Const.HP].get("sim_gate_trainable", True)
        binning_type = model_cfg[Const.HP].get("binning_type", 'default')

        if num_bins <= 0:
            raise ValueError(
                "num_bins must be positive"
            )
        if embedding_size <= 0:
            raise ValueError(
                "embedding_size must be positive"
            )
        if gru_units is not None:
            if gru_units <= 0:
                raise ValueError(
                    "gru_units must be positive if provided"
                )

        self.num_bins = num_bins
        self.embedding_size = embedding_size
        self.interval = 2.0 / num_bins
        self.binning_type = binning_type
        self.gru_units = gru_units

        # 分桶 embedding
        self.sim_bins = nn.Embedding(num_bins, embedding_size)

        # 可训练参数 w 和 b
        self.sim_gate_w = nn.Parameter(torch.tensor([sim_gate_w], dtype=torch.float32),
                                       requires_grad=sim_gate_trainable)
        self.sim_gate_b = nn.Parameter(torch.tensor([sim_gate_b], dtype=torch.float32),
                                       requires_grad=sim_gate_trainable)

        # GRU（可选）
        self.gru = nn.GRU(embedding_size, gru_units, batch_first=True) if gru_units is not None else None

    def sim_gate_func(self, cos_sim, w, b):
        """ 相似门函数: sigmoid(w*cos - b) / sigmoid(w - b) """
        return torch.sigmoid(w * cos_sim - b) / torch.sigmoid(w - b)

    def forward(self, target, sequence, mask=None):
        """
        执行SRN模块前向传播

        参数:
            target: [B, 1, E]   - 目标向量
            sequence: [B, L, E] - 历史行为序列
            mask: [B, L] or None - 序列mask，True表示保留

        返回:
            [B, E] 或 [B, E + gru_units]
        """

        B, L, E = sequence.shape
        B, num_rerank, E = target.shape

        # Cosine similarity [B, L]
        target_norm = F.normalize(target, dim=-1)  # [B, 1, E]
        sequence_norm = F.normalize(sequence, dim=-1)  # [B, L, E]
        if torch.onnx.is_in_onnx_export() or not self.training:
            target_norm = target_norm.unsqueeze(2)
            sequence_norm = sequence_norm.unsqueeze(1)
        cos_sim = torch.sum(target_norm * sequence_norm, dim=-1)  # [B, L]

        # Normalize 分桶方式
        if self.binning_type == 'normalize':
            if mask is not None:
                cos_sim_masked = cos_sim.masked_fill(~mask, -1e9)
                max_val = cos_sim_masked.max(dim=1, keepdim=True).values
                min_val = cos_sim_masked.masked_fill(cos_sim_masked == -1e9, 1e9).min(dim=1, keepdim=True).values
            else:
                max_val = cos_sim.max(dim=1, keepdim=True).values
                min_val = cos_sim.min(dim=1, keepdim=True).values
            cos_sim = 2.0 * (cos_sim - min_val) / (max_val - min_val + 1e-6) - 1.0

        # 映射到分桶 [B, L]
        cos_sim_idx = ((cos_sim + 1) / self.interval).long()

        cos_sim_idx = torch.clamp(cos_sim_idx, 0, self.num_bins - 1)

        # 查表获得每个位置对应的bin embedding
        bin_embed = self.sim_bins(cos_sim_idx)  # [B, L, E]

        # 应用 mask（zero-out）
        if mask is not None:
            if torch.onnx.is_in_onnx_export() or not self.training:
                bin_embed = bin_embed * mask.unsqueeze(1).unsqueeze(-1).float()
            else:
                bin_embed = bin_embed * mask.unsqueeze(-1).float()

        # 应用相似门函数
        sim_weight = self.sim_gate_func(cos_sim, self.sim_gate_w, self.sim_gate_b)  # [B, L]
        sim_weight = sim_weight.unsqueeze(-1)  # [B, L, 1]
        bin_embed = bin_embed * sim_weight  # [B, L, E]

        # 平均池化作为兴趣向量
        if mask is not None:
            valid_count = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1]
            if torch.onnx.is_in_onnx_export() or not self.training:
                valid_count = valid_count.unsqueeze(1)  # [B, 256, 1]
                interest_embed = bin_embed.sum(dim=2) / valid_count  # [B, 256, E]
            else:
                interest_embed = bin_embed.sum(dim=1) / valid_count  # [B, E]
        else:
            if torch.onnx.is_in_onnx_export() or not self.training:
                interest_embed = bin_embed.mean(dim=2)  # [B, E]
            else:
                interest_embed = bin_embed.mean(dim=1)  # [B, E]

        # 可选GRU兴趣演化建模
        if self.gru is not None:
            if mask is not None:
                lengths = mask.sum(dim=1).cpu()
                packed_seq = nn.utils.rnn.pack_padded_sequence(bin_embed,
                                                               lengths, batch_first=True, enforce_sorted=False)
                _, h_n = self.gru(packed_seq)
            else:
                _, h_n = self.gru(bin_embed)  # h_n: [1, B, H]
            gru_out = h_n.squeeze(0)  # [B, H]
            output = torch.cat([interest_embed, gru_out], dim=-1)  # [B, E+H]
        else:
            output = interest_embed  # [B, E]

        return output


@ModelRegistry.register(opt_subs={"BaseAttention"})
class BaseAttention(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        self.num_heads = model_cfg[Const.HP].get("num_heads", None)
        self.project_query = model_cfg[Const.HP].get("project_query", False)
        self.output_type = model_cfg[Const.HP].get("output_type", "sum")
        self.use_bias = model_cfg[Const.HP].get("use_bias", False)
        self.query_activation = model_cfg[Const.HP].get("activation", nn.Tanh())
        self.kernel_initializer = model_cfg[Const.HP].get("kernel_initializer", 'glorot_uniform')
        self.return_attention_probs = model_cfg[Const.HP].get("return_attention_probs", False)

        self.query_project_layer = None
        self.query_mapper = None
        self.key_mapper = None
        self.value_mapper = None

        self.norm = model_cfg[Const.HP].get("norm", None)
        dropout_rate = model_cfg[Const.HP].get("dropout_rate", 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_activation = model_cfg[Const.HP].get("output_activation", None)
        self.orders = model_cfg[Const.HP].get("orders", "adn")
        self.dot_attn = model_cfg[Const.HP].get("dot_attn", False)

        query_size = model_cfg[Const.HP].get("query_size", 16)
        key_size = model_cfg[Const.HP].get("key_size", 16)
        value_size = model_cfg[Const.HP].get("value_size", 16)
        if self.project_query:
            self.query_project_layer = nn.Linear(query_size, key_size, bias=self.use_bias)
        self.query_mapper = nn.Linear(query_size, key_size, bias=self.use_bias)
        self.key_mapper = nn.Linear(key_size, key_size, bias=self.use_bias)
        self.value_mapper = nn.Linear(value_size, value_size, bias=self.use_bias)

    def _compute_attention_score(self, query, key):
        key_dim = query.size(-1)
        score = torch.matmul(query, key.transpose(-2, -1))  # [batch, heads, query_len, value_len]
        return score / math.sqrt(key_dim)

    def forward(self, query, key, mask=None):
        value = key

        query_mask = mask[0] if mask else None
        key_mask = mask[1] if mask else None

        if len(query.shape) == 2:
            query = query.unsqueeze(1)
            is_2d_query = True
        else:
            is_2d_query = False

        if self.project_query:
            query = self.query_project_layer(query)
            if self.query_activation:
                query = self.query_activation(query)
        if self.num_heads:
            query = self._transpose_qkv(self.query_mapper(query))
            key = self._transpose_qkv(self.key_mapper(key))
            value = self._transpose_qkv(self.value_mapper(value))
        else:
            query = self.query_mapper(query)
            key = self.key_mapper(key)
            value = self.value_mapper(value)

        if self.dot_attn:
            attn_scores = self._compute_attention_score(query, key)
        else:
            attn_scores = torch.matmul(query, key.transpose(-2, -1))

        if key_mask is not None:
            if self.dot_attn:
                key_mask = self._adapt_mask(key_mask, attn_scores.shape)
                attn_scores = attn_scores.masked_fill(~key_mask, float('-inf'))
            else:
                bsz, tgt_len, src_len = attn_scores.shape
                key_mask = key_mask.unsqueeze(1)
                key_mask = key_mask.expand(bsz, tgt_len, src_len)
                attn_scores = attn_scores.masked_fill(~key_mask, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)

        if self.output_type == "concat":
            outputs = attn_probs.unsqueeze(-1) * value.unsqueeze(-3)
            outputs = outputs.view(*outputs.shape[:-2], -1)
        else:
            outputs = torch.matmul(attn_probs, value)

        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1)
            if self.num_heads:
                query_mask = query_mask.unsqueeze(1)
            outputs *= query_mask.type_as(outputs)

        if self.num_heads:
            outputs = self._transpose_output(outputs)

        if is_2d_query:
            outputs = outputs.squeeze(1)
            attn_probs = attn_probs.squeeze(2)

        outputs = self._apply_output_processing(outputs)

        if self.return_attention_probs:
            return attn_probs, outputs
        return outputs

    def _apply_output_processing(self, outputs):
        if self.norm:
            outputs = self.norm(outputs)
        if self.output_activation:
            outputs = self.output_activation(outputs)
        outputs = self.dropout(outputs)
        return outputs

    def _transpose_qkv(self, x):
        bsz, seq_len, dim = x.size()
        head_dim = dim // self.num_heads
        x = x.view(bsz, seq_len, self.num_heads, head_dim).transpose(1, 2)
        return x

    def _transpose_output(self, x):
        bsz, heads, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, heads * head_dim)

    def _adapt_mask(self, mask, attn_shape):
        bsz, _, tgt_len, src_len = attn_shape
        mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, src_len]
        return mask.expand(bsz, self.num_heads, tgt_len, src_len)


class Expert(torch.nn.Module):
    def __init__(self, hidden_dims=None):
        self.hidden_dims = hidden_dims
        super(Expert, self).__init__()
        self.net = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)])

    def forward(self, x):
        n_net = len(self.net)
        for i in range(n_net):
            x = self.net[i](x)
        return x


@ModelRegistry.register(opt_subs={"MMOE"})
class MMOE(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        self.num_tasks = model_cfg[Const.HP].get("num_tasks")
        self.stack_outputs = model_cfg[Const.HP].get("stack_outputs", False)
        self.topk = model_cfg[Const.HP].get("topk", None)
        self.dselectk_strategy = model_cfg[Const.HP].get("dselectk_strategy", "task_level_routing")
        self.gate_activation = model_cfg[Const.HP].get("gate_activation", "softmax")
        self.use_gate_bias = model_cfg[Const.HP].get("use_gate_bias", True)
        self.orders = model_cfg[Const.HP].get("orders", "adn")

        activation = model_cfg[Const.HP].get("activation", None)
        norm = model_cfg[Const.HP].get("norm", None)
        dropout_rate = model_cfg[Const.HP].get("dropout_rate", 0.0)
        self.activation_fn = getattr(F, activation) if isinstance(activation, str) else None
        self.norm = nn.LayerNorm(norm) if norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        expert_config = model_cfg[Const.HP].get("expert_config", None)
        num_experts = model_cfg[Const.HP].get("num_experts", None)
        input_dim = model_cfg[Const.HP].get("input_dim")
        use_gate_bias = model_cfg[Const.HP].get("use_gate_bias", True)
        self.out_dim = expert_config["hidden_dims"][-1]

        # === 构建专家层 ===
        self.expert_layers = nn.ModuleList()
        if isinstance(expert_config, list):
            if num_experts is not None and len(expert_config) != num_experts:
                logging.info(f"[Warning] num_experts={num_experts} \
                             is overwritten by expert_config length {len(expert_config)}")
            self.num_experts = len(expert_config)
            for cfg in expert_config:
                self.expert_layers.append(self._build_expert(cfg, input_dim))
        else:
            self.num_experts = num_experts
            for _ in range(self.num_experts):
                self.expert_layers.append(self._build_expert(expert_config, input_dim))

        # === 构建门控层 ===
        self.gate_layers = nn.ModuleList([
            nn.Linear(input_dim, self.num_experts, bias=use_gate_bias)
            for _ in range(self.num_tasks)
        ])

    def _build_expert(self, cfg: dict, input_dim: int):
        layers = []
        last_dim = input_dim
        for dim in cfg.get("hidden_dims", []):
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.ReLU())
            last_dim = dim
        return nn.Sequential(*layers)

    def _apply_orders(self, x):
        for op in self.orders:
            if op == 'n' and self.norm:
                x = self.norm(x)
            elif op == 'd' and self.dropout:
                x = self.dropout(x)
            elif op == 'a' and self.activation_fn:
                x = self.activation_fn(x)
        return x

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        expert_outputs = [expert(x) for expert in self.expert_layers]  # [B, D] * num_experts
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, E, D]
        E, D = expert_outputs.size(1), expert_outputs.size(2)

        outputs = []
        for gate_layer in self.gate_layers:
            gate_logits = gate_layer(x)  # [B, E]
            if self.gate_activation == "softmax":
                gate_weights = F.softmax(gate_logits, dim=-1)
            elif self.gate_activation == "sigmoid":
                gate_weights = torch.sigmoid(gate_logits)
            else:
                gate_weights = gate_logits  # linear

            gate_weights = gate_weights.unsqueeze(1)  # [B, 1, E]
            output = torch.bmm(gate_weights, expert_outputs)  # [B, 1, D]
            output = output.squeeze(1)  # [B, D]
            outputs.append(self._apply_orders(output))  # norm+dropout+activation

        if self.stack_outputs:
            return torch.stack(outputs, dim=1)  # [B, T, D]
        else:
            return outputs  # list of [B, D]


@ModelRegistry.register()
class AverageLayer(BaseModel):
    """
    求平均层，用于在指定轴上求平均值降维（支持带mask的求平均）

    :param axis: 所要求平均的轴
    :param keep_dims: 结果是否和输入保持同一维度
    :param mask: 是否执行mask。如果为 True，那么只会取非mask部分的均值作为计算结果
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.axis = model_cfg[Const.HP].get("axis", None)
        self.keep_dims = model_cfg[Const.HP].get("keep_dims", False)
        self.mask_flag = model_cfg[Const.HP].get("mask", True)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mask_flag and mask is not None:
            expanded_mask = mask.unsqueeze(-1).type_as(inputs)  # [batch, seq_len, 1]
            expanded_mask = expanded_mask.expand_as(inputs)  # [batch, seq_len, emb_dim]
            masked_inputs = inputs * expanded_mask

            # 求和 & 非零mask数量
            sum_inputs = torch.sum(masked_inputs, dim=self.axis, keepdim=self.keep_dims)
            mask_count = torch.sum(expanded_mask, dim=self.axis, keepdim=self.keep_dims)
            mask_count = mask_count + 1e-9  # 避免除以0

            return sum_inputs / mask_count

        # 不带mask的平均
        return torch.mean(inputs, dim=self.axis, keepdim=self.keep_dims)
