import abc
import logging
from typing import Dict, List, Tuple, Optional, Callable, Union, OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from modeling.generic.initialization import truncated_normal
from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import FeatConst



def get_activation(act_func: str):
    """获取激活函数模块"""
    if act_func == "relu":
        return nn.ReLU()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation function: {act_func}")


def get_norm_layer(norm_type: str, dim: int) -> Optional[nn.Module]:
    """ 获取归一化层 """
    if norm_type == 'batch_norm':
        return nn.BatchNorm1d(dim)
    elif norm_type == 'layer_norm':
        return nn.LayerNorm(dim)
    elif norm_type is None or norm_type == '':
        return None
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")


class CrossNetwork(nn.Module):

    def __init__(self, input_dim, num_cross_layers):
        super().__init__()
        self.num_cross_layers = num_cross_layers

        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim))
            for _ in range(num_cross_layers)
        ])  # list of [cross_input_dim]

        self.cross_bias = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim)) for _ in range(num_cross_layers)]
        )  # list of [cross_input_dim]

    def forward(self, x):
        """
        x: [batch_size, cross_input_dim]
        对每个位置的 cross_input_dim 做 Cross 操作
        FP16下x_0*xlw点积易溢出，需FP32保护
        """
        x_0 = x
        x_l = x_0
        # Cross的x_0*xlw点积在FP16下极易溢出，整段升FP32
        if x_0.dtype == torch.float16:
            x_0 = x_0.float()
            x_l = x_l.float()
            for i in range(self.num_cross_layers):
                xlw = F.linear(x_l, self.cross_w[i].unsqueeze(0).float())
                x_l = x_0 * xlw + self.cross_bias[i].float() + x_l
            return x_l.to(x.dtype)
        for i in range(self.num_cross_layers):
            xlw = F.linear(x_l, self.cross_w[i].unsqueeze(0))
            x_l = x_0 * xlw + self.cross_bias[i] + x_l
        return x_l


class MLPLayer(nn.Module):

    def __init__(self,
                 hidden_layers: List,
                 act_func: str = 'relu',
                 dropout_rate: float = 0.0,
                 use_bn: bool = False):
        super().__init__()

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
            if dropout_rate > 0.0:
                layers.append(torch.nn.Dropout(dropout_rate))
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if use_bn:
                layers.append(torch.nn.BatchNorm1d(out_dim))
            if act_func and i != (len(hidden_layers) - 1):
                layers.append(get_activation(act_func))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor):
        x = inputs
        for layer in self.layers:
            # 由于输入均是[bs, num_rerank, emb_dim],bn要求中间维是emb_dim,因此需要如下处理
            if isinstance(layer, nn.BatchNorm1d):
                if x.dim() == 3:
                    x = x.transpose(1, 2)  # (B, L, C) => (B, C, L)
                    x = layer(x)
                    x = x.transpose(1, 2)  # (B, C, L) => (B, L, C)
                else:
                    x = layer(x)
            else:
                x = layer(x)
        return x


class PPNetLayer(nn.Module):
    """PPNet 模块的 PyTorch 实现

    Args:
        feature_emb_dim (int): 特征嵌入维度
        gate_emb_dim (int): 门控嵌入维度
        mlp_layer_dims (list): MLP层部分的各个层输出维度
        gate_layer_dims (list): 各个gate层的中间维度
        use_bias (bool): 是否使用偏置，默认为 False
        act_func (str): 激活函数类型，默认为 "relu"
        is_act (bool): 是否对MLP使用激活函数，默认为 True
        batch_norm (bool): 是否使用 batch normalization，默认为 False
        dropout_rate (float): dropout 概率，默认为 0.1
        is_stop_grad (bool): 是否截断输入 feature_emb 的梯度，默认为 True
    """

    def __init__(self, feature_emb_dim: int, gate_emb_dim: int,
                 mlp_layer_dims: List[int], gate_layer_dims: List[int],
                 use_bias: bool = False, act_func: str = "relu",
                 is_act: bool = True, batch_norm: bool = False,
                 dropout_rate: float = 0.1, is_stop_grad: bool = True):
        super(PPNetLayer, self).__init__()

        if len(mlp_layer_dims) != len(gate_layer_dims):
            raise ValueError("mlp_layer_dims and gate_layer_dims must have the same length")

        self.feature_emb_dim = feature_emb_dim
        self.gate_emb_dim = gate_emb_dim
        self.use_bias = use_bias
        self.is_stop_grad = is_stop_grad
        self.mlp_layer_dims = mlp_layer_dims
        self.gate_layer_dims = gate_layer_dims
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.is_act = is_act
        self.act_func = act_func

        self.activation = get_activation(act_func)

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None

        self.gate_out_dims = [feature_emb_dim] + mlp_layer_dims[:-1]

        self.gates = nn.ModuleList()
        for _, (hidden_dim, out_dim) in enumerate(zip(gate_layer_dims, self.gate_out_dims)):
            gate_input_dim = feature_emb_dim + gate_emb_dim
            gate = nn.Sequential(
                nn.Linear(gate_input_dim, hidden_dim),
                get_activation(act_func),
                nn.Linear(hidden_dim, out_dim),
                nn.Sigmoid()  # 论文中 gate 输出使用 sigmoid
            )
            self.gates.append(gate)

        # 创建 MLP 层
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # 第一层输入维度是特征嵌入维度
        input_dim = feature_emb_dim

        for _, output_dim in enumerate(mlp_layer_dims):
            layer = nn.Linear(input_dim, output_dim, bias=use_bias)
            self.layers.append(layer)

            # 添加 BatchNorm
            if batch_norm:
                self.bn_layers.append(nn.LayerNorm(output_dim))

            input_dim = output_dim

    def forward(self, inputs: List[torch.Tensor]):
        """
        Args:
            inputs (list): 包含两个张量 [feature_emb, gate_emb]
        Returns:
            torch.Tensor: 网络输出
        """
        feature_emb, gate_emb = inputs

        # 验证输入维度
        if feature_emb.size(-1) != self.feature_emb_dim:
            raise ValueError(f"Expected feature_emb dim {self.feature_emb_dim}, got {feature_emb.size(-1)}")
        if gate_emb.size(-1) != self.gate_emb_dim:
            raise ValueError(f"Expected gate_emb dim {self.gate_emb_dim}, got {gate_emb.size(-1)}")

        # 梯度截断
        if self.is_stop_grad:
            stop_feature_emb = feature_emb.detach()
        else:
            stop_feature_emb = feature_emb

        # 拼接 gate 输入
        gate_input = torch.cat([stop_feature_emb, gate_emb], dim=-1)

        hidden_out = feature_emb
        for i, (gate, layer) in enumerate(zip(self.gates, self.layers)):
            gate_out = 2 * gate(gate_input)  # 论文中 gate 输出乘以 2

            hidden_out = hidden_out * gate_out
            hidden_out = layer(hidden_out)
            if self.batch_norm:
                hidden_out = self.bn_layers[i](hidden_out)

            if self.is_act and i != (
                    len(self.layers) - 1):
                hidden_out = self.activation(hidden_out)

            if self.dropout and i != (len(self.layers) - 1):
                hidden_out = self.dropout(hidden_out)

        return hidden_out


class Dense(nn.Module):
    """
    Dense模块的Pytorch实现
    """

    def __init__(self, units, kernel_initializer='he_uniform', activation=None,
                 use_bias=True, bias_initializer='zeros'):
        super(Dense, self).__init__()
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.linear = None

    def forward(self, inputs):
        """前向传播"""
        if self.linear is None:
            # 动态创建线性层
            input_dim = inputs.shape[-1]
            self.linear = nn.Linear(input_dim, self.units, bias=self.use_bias)

            # 初始化权重和偏置
            if self.kernel_initializer == 'he_uniform':
                nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
            elif self.kernel_initializer == 'glorot_uniform':
                nn.init.xavier_uniform_(self.linear.weight)
            elif self.kernel_initializer == 'normal':
                nn.init.normal_(self.linear.weight)
            else:
                nn.init.xavier_uniform_(self.linear.weight)

            if self.use_bias and self.linear.bias is not None:
                if self.bias_initializer == 'zeros':
                    nn.init.zeros_(self.linear.bias)
                else:
                    nn.init.zeros_(self.linear.bias)

            # 确保在相同设备上
            self.linear = self.linear.to(inputs.device)

        output = self.linear(inputs)

        # 激活函数
        if self.activation is not None:
            if self.activation == 'relu':
                output = F.relu(output)
            elif self.activation == 'sigmoid':
                output = torch.sigmoid(output)
            elif self.activation == 'tanh':
                output = torch.tanh(output)
            elif self.activation == 'softmax':
                output = F.softmax(output, dim=-1)
            elif self.activation == 'linear':
                pass
            else:
                raise ValueError(f"Unsupported activation: {self.activation}")

        return output


class DomainGateModule(nn.Module):
    def __init__(self, units=90, kernel_initializer='he_uniform', activation='relu', use_flatten=False):
        super(DomainGateModule, self).__init__()
        # domain gate dense layer
        self.domain_gate_dense = Dense(
            units=units,
            kernel_initializer=kernel_initializer,
            activation=activation
        )
        self.use_flatten = use_flatten
        if use_flatten:
            self.flatten = nn.Flatten()

    def forward(self, gate_feature_embedding, target_feature_embedding):
        # domain gate dense
        dense_output = self.domain_gate_dense(gate_feature_embedding)
        # domain gate sigmoid: 2 * sigmoid(x)
        gate_weights = 2 * torch.sigmoid(dense_output)
        # domain multiply (element-wise multiplication)
        weighted_embedding = gate_weights * target_feature_embedding

        # optional flatten
        if self.use_flatten:
            weighted_embedding = self.flatten(weighted_embedding)

        return weighted_embedding


class BaselinePPNet(nn.Module):
    def __init__(self,
                 hidden_dims: List[int],
                 domain_gate_input_dim: int,
                 domain_no_target_input_dim: int,
                 target_input_dim: int,
                 domain_gate_units: int,
                 num_cross_layers: int,
                 dropout_rate: float = 0.1):
        super(BaselinePPNet, self).__init__()

        if len(hidden_dims) != 4:
            raise ValueError("BaselinePPNet expects 4 hidden dims to match baseline POSO branches")

        self.hidden_dims = hidden_dims
        self.domain_gate_input_dim = domain_gate_input_dim
        self.domain_no_target_input_dim = domain_no_target_input_dim
        self.target_input_dim = target_input_dim
        self.domain_gate_units = domain_gate_units
        self.output_dim = domain_no_target_input_dim + hidden_dims[-1]

        self.domain_gate_dense = nn.Linear(domain_gate_input_dim, domain_gate_units)

        self.poso_gate_dense_1 = nn.Linear(target_input_dim, hidden_dims[0])
        # Keep branch 2/3 modules for baseline graph/checkpoint compatibility; forward only uses active branches.
        self.poso_gate_dense_2 = nn.Linear(target_input_dim, hidden_dims[1])
        self.poso_gate_dense_3 = nn.Linear(target_input_dim, hidden_dims[2])
        self.poso_gate_dense_4 = nn.Linear(target_input_dim, hidden_dims[3])

        self.mlp_1 = nn.Linear(domain_no_target_input_dim, hidden_dims[0])
        self.mlp_2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.mlp_3 = nn.Linear(hidden_dims[0], hidden_dims[2])
        self.mlp_4 = nn.Linear(hidden_dims[0], hidden_dims[3])

        self.norm_dropout_dnn_output = NormDropoutLayer(
            in_dim=hidden_dims[-1],
            norm_type='batch_norm',
            dropout_rate=dropout_rate,
            order='nd'
        )

        self.cross_network = CrossNetwork(
            input_dim=domain_no_target_input_dim,
            num_cross_layers=num_cross_layers
        )

        self.lhuc_gate_dense = nn.Linear(target_input_dim, self.output_dim)

    @staticmethod
    def _baseline_dense(linear: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(linear(inputs))

    def forward(self,
                concat_domain_embedding: torch.Tensor,
                concat_domain_no_target_embedding: torch.Tensor,
                domain_type1_embedding: torch.Tensor) -> torch.Tensor:
        """
        concat_domain_embedding: [B, C, D_gate]
        concat_domain_no_target_embedding: [B, C, N, D_emb]
        domain_type1_embedding: [B, C, D_target]
        """
        if concat_domain_no_target_embedding.dim() != 4:
            raise ValueError("concat_domain_no_target_embedding must keep feature axis: [B, C, N, D]")

        domain_gate = 2 * torch.sigmoid(self._baseline_dense(self.domain_gate_dense, concat_domain_embedding))
        gated_embedding = concat_domain_no_target_embedding * domain_gate.unsqueeze(-2)
        gated_embedding = gated_embedding.flatten(start_dim=-2)

        poso_gate_1 = 2 * torch.sigmoid(self._baseline_dense(self.poso_gate_dense_1, domain_type1_embedding))
        poso_gate_4 = 2 * torch.sigmoid(self._baseline_dense(self.poso_gate_dense_4, domain_type1_embedding))

        mlp_multiply_1 = self._baseline_dense(self.mlp_1, gated_embedding) * poso_gate_1
        mlp_multiply_4 = self._baseline_dense(self.mlp_4, mlp_multiply_1) * poso_gate_4
        final_mlp_output = self.norm_dropout_dnn_output(mlp_multiply_4)

        cross_output = self.cross_network(gated_embedding)

        lhuc_weights = 2 * torch.sigmoid(self._baseline_dense(self.lhuc_gate_dense, domain_type1_embedding))
        final_mlp_weights, cross_weights = torch.split(
            lhuc_weights,
            [final_mlp_output.size(-1), cross_output.size(-1)],
            dim=-1
        )
        return torch.cat([final_mlp_output * final_mlp_weights, cross_output * cross_weights], dim=-1)


class NormDropoutLayer(nn.Module):
    """ 封装每个 DNN hidden 层执行顺序（Norm / Dropout / Activation） """

    def __init__(
        self,
        in_dim: int,
        norm_type: str = None,
        dropout_rate: float = 0.0,
        activation: str = None,
        order: str = "nad"  # 默认顺序 Norm -> Activation -> Dropout
    ):
        super().__init__()
        self.in_dim = in_dim
        self.norm_type = norm_type
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.order = order

        # 构造组件
        self.norm_layer = get_norm_layer(norm_type, in_dim)
        self.dropout_layer = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.activation_layer = get_activation(activation) if activation else None

        if not all(c in "nad" for c in order):
            raise ValueError("Order must be a combination of 'n', 'a', 'd', e.g., 'nad'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.order:
            if op == "n":
                if self.norm_layer is not None:
                    if x.dim() == 3:
                        x = x.transpose(1, 2)  # (B, L, C) => (B, C, L)
                        x = self.norm_layer(x)
                        x = x.transpose(1, 2)  # (B, C, L) => (B, L, C)
                    else:
                        x = self.norm_layer(x)
            elif op == "a":
                if self.activation_layer is not None:
                    x = self.activation_layer(x)
            elif op == "d":
                if self.dropout_layer is not None:
                    x = self.dropout_layer(x)
        return x


class DeepNetwork(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],  # 包含从输入到输出层所有维度
        hidden_activations: Union[str, List[str]] = "relu",
        output_activation: Optional[str] = None,
        dropout_rates: Union[float, List[float]] = 0.0,
        norms: Union[str, List[str]] = None,
        orders: Union[str, List[str]] = "nad",
        use_bias: bool = True,
        l2_reg: float = 1e-5,
        return_hidden_outputs: bool = False,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.l2_reg = l2_reg
        self.return_hidden_outputs = return_hidden_outputs
        self.num_layers = len(hidden_dims) - 1  # 不包括输入
        self.hidden_activations = hidden_activations
        self.dropout_rates = dropout_rates
        self.norms = norms
        self.orders = orders

        # 支持 list 或统一值
        self._expand_list_like_args()

        layers = []

        for i in range(self.num_layers):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i+1]

            # Linear Layer
            linear_layer = torch.nn.Linear(in_dim, out_dim)
            layers.append(linear_layer)

            # 若非最后一层才添加 norm/dropout/activation
            if i < self.num_layers - 1:
                normdroplayer = NormDropoutLayer(
                    in_dim=out_dim,  # norm 是作用在输出上的
                    norm_type=self.norms[i],
                    dropout_rate=self.dropout_rates[i],
                    activation=self.hidden_activations[i],
                    order=self.orders[i]
                )
                layers.append(normdroplayer)
            # 最后一层可能有 activation
            else:
                if output_activation:
                    act_layer = get_activation(output_activation)
                    layers.append(act_layer)

        self.model = nn.Sequential(*layers)

    def _expand_list_like_args(self):
        """ 保证每个参数扩展为 list 且长度等于层数 """
        self.hidden_activations = self._to_list(self.hidden_activations, self.num_layers - 1)
        self.dropout_rates = self._to_list(self.dropout_rates, self.num_layers - 1)
        self.norms = self._to_list(self.norms, self.num_layers - 1)
        self.orders = self._to_list(self.orders, self.num_layers - 1)

    def _to_list(self, val, length):
        if not isinstance(val, list):
            return [val] * length
        elif len(val) != length:
            raise ValueError(f"List length mismatch: expected {length}, got {len(val)}")
        return val

    def forward(self, inputs: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        outputs = []
        cur_tensor = inputs

        for name, layer in self.model.named_children():
            cur_tensor = layer(cur_tensor)
            # 注意只能是中间层记录 hidden 输出，不要包括 output_activation
            if self.return_hidden_outputs and name != "output_activation":
                outputs.append(cur_tensor)

        if self.return_hidden_outputs:
            return outputs
        else:
            return cur_tensor

    def get_regularization_loss(self) -> torch.Tensor:
        reg_loss = 0.0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # weight**2 sum在FP16下可能溢出，upcast到FP32
                w = module.weight
                reg_loss += torch.sum(w.float() ** 2)
        return self.l2_reg * reg_loss
