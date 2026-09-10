import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import OrderedDict
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.initialization import truncated_normal
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
from utils.common_utils import compute_user_item_feature_dims
import logging


@ModelRegistry.register()
class DLRModule(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)
        self.dlrm_modules = nn.ModuleList([
            self.init_sub_model(sub_key)
            for sub_key in model_cfg[Const.SUB_MODELS].keys()
        ])

    def forward(self,
                past_ids,
                num_rerank,
                model_inputs,
                user_feature_embs_original,
                item_feature_embs_original,
                seq_feature_embs):
        init_dlrm_output = None
        init_sparsity, init_loss = 0.0, 0.0
        for dlrm_module in self.dlrm_modules:
            dlrm_output = dlrm_module(past_ids=past_ids, num_rerank=num_rerank, model_inputs=model_inputs,
                                      user_feature_embs_original=user_feature_embs_original,
                                      item_feature_embs_original=item_feature_embs_original,
                                      seq_feature_embs=seq_feature_embs)
            if init_dlrm_output is None:
                init_dlrm_output = dlrm_output["deep_outputs"]
                init_loss, init_sparsity = dlrm_output["deep_loss"]
            else:
                init_dlrm_output = init_dlrm_output + dlrm_output["deep_outputs"]
                init_loss, init_sparsity = dlrm_output["deep_loss"][0] + \
                                           init_loss, dlrm_output["deep_loss"][1] + init_sparsity

        return {"deep_outputs": init_dlrm_output, "deep_loss": (init_loss, init_sparsity)}


@ModelRegistry.register()
class RankMixingInput(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        dim_per_token = model_cfg[Const.HP].get("dim_per_token")
        x_dim = model_cfg[Const.HP].get("x_dim")
        k = model_cfg[Const.HP].get("k")
        k_multiplier = model_cfg[Const.HP].get("k_multiplier", 1)
        if (x_dim % dim_per_token) != 0:
            raise ValueError("x_dim must be devide by dim per split")
        T = x_dim // dim_per_token
        num_heads = T
        inner_dim = int(num_heads * k * k_multiplier)
        if (inner_dim % num_heads) != 0:
            raise ValueError("inner_dim must be devide by num_heads")
        self.x_dim = x_dim
        self.num_heads = num_heads
        self.dim_per_token = dim_per_token
        self.inner_dim = inner_dim
        self.proj = torch.nn.Linear(dim_per_token, inner_dim, bias=False)

    def forward(self, x):
        # 输入x是所有用户、商品、序列特征拼接而成的特征向量，形状为(B, \sum_{D_e}), D_e是不同特征的长度
        B, D = x.size()
        if self.x_dim != D:
            raise ValueError("x_dim must be same with input dim")
        x = x[:, :self.num_heads * self.dim_per_token]
        # 形状（B, T, inner_dim）
        proj_x = self.proj(x.view(B, self.num_heads, self.dim_per_token))
        return proj_x


@ModelRegistry.register()
class TokenMxing(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        RankMixer中的TokenMixing模块,
        对应原论文公式2,3,4,5
        dim_per_token：公式2里的d，即重新划分的token的维度
        x_dim: 所有特征拼接起来的维度总和
        为避免信息损失，要求x_dim可以被dim_per_token整除。
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        dim_per_token = model_cfg[Const.HP].get("dim_per_token")
        x_dim = model_cfg[Const.HP].get("x_dim")
        k = model_cfg[Const.HP].get("k")
        k_multiplier = model_cfg[Const.HP].get("k_multiplier")
        T = x_dim // dim_per_token
        num_heads = T
        inner_dim = int(num_heads * k * k_multiplier)
        self.num_heads = num_heads
        self.ln = torch.nn.LayerNorm((inner_dim,), eps=1e-7)

    def forward(self, x):
        B = x.size(0)
        tm_x = torch.permute(x, (0, 2, 1)).reshape(B, self.num_heads, -1)
        # 形状（B, num_heads, inner_dim）
        return self.ln(x + tm_x)


@ModelRegistry.register()
class PerTokenFFN(BaseModel):
    """
    Per-token position-wise MLP with configurable depth L.
    Each token position t has its own stack of Linear layers.

    For L layers:
      layer 1: D -> kD
      layer 2..L-1: kD -> kD
      layer L: kD -> D
    Activations: GELU after layers 1..L-1 (no activation on last).
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        D = model_cfg[Const.HP].get("D")
        T = model_cfg[Const.HP].get("T")
        k = model_cfg[Const.HP].get("k")
        num_layers = model_cfg[Const.HP].get("num_layers")
        bias = model_cfg[Const.HP].get("bias")
        dropout_p = model_cfg[Const.HP].get("dropout_p")

        self.D = D
        self.T = T
        self.kD = int(round(k * D))
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # Build per-position weights for each layer
        in_dims = [D] + [self.kD] * (num_layers - 1)
        out_dims = [self.kD] * (num_layers - 1) + [D]
        # out_dims length is num_layers; first num_layers-1 are kD, last is D

        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(T, din, dout))
            for din, dout in zip(in_dims, out_dims)
        ])
        if bias:
            self.b = nn.ParameterList([
                nn.Parameter(torch.empty(T, dout))
                for dout in out_dims
            ])
        else:
            self.b = None

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_layers):
            nn.init.xavier_normal_(self.W[i])
            if self.b is not None:
                nn.init.zeros_(self.b[i])

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (bs, T, D)  ->  v: (bs, T, D)
        """
        bs, T, D = s.shape

        x = s
        # Apply layers 0..L-2 with GELU, final layer without activation
        for i in range(self.num_layers):
            x = torch.einsum("bti,tid->btd", x, self.W[i])
            if self.b is not None:
                x = x + self.b[i]
            if i < self.num_layers - 1:
                x = F.gelu(x)
                x = self.dropout(x)
        # optional dropout on output (comment out if you want *exact* equations)
        x = self.dropout(x)
        return x


@ModelRegistry.register(req_subs={"PerTokenFFN"})
class SparseMoE(BaseModel):
    """
    ReLU-Routed Sparse MoE with per-token experts.

    Given router h(·) and experts e_j(·):
        G_{i,j} = ReLU(h(s_i))
        v_i     = sum_{j=1..N_e} G_{i,j} * e_j(s_i)

    Args:
        num_experts_per_token (int): N_e, number of experts per token.
        inner_dim (int): D, hidden size per token.
        num_tokens (int): T, number of token positions.
        k, bias, dropout_p: forwarded to PerTokenFFN.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        num_experts_per_token = model_cfg[Const.HP].get("num_experts_per_token")
        inner_dim = model_cfg[Const.HP].get("inner_dim")
        num_tokens = model_cfg[Const.HP].get("num_tokens")
        k = model_cfg[Const.HP].get("k")
        num_layers_per_expert = model_cfg[Const.HP].get("num_layers_per_expert")
        bias = model_cfg[Const.HP].get("bias")
        dropout_p = model_cfg[Const.HP].get("dropout_p")

        self.D = inner_dim
        self.T = num_tokens
        self.Ne = num_experts_per_token
        self.relu_threshold = 1e-4

        # Router h(·): shared across positions, maps R^D -> R^{N_e}
        self.router = nn.Linear(self.D, self.Ne, bias=False)

        # Experts: N_e copies of PerTokenFFN (each is position-specific over T)
        self.model_cfg[Const.SUB_MODELS]["PerTokenFFN"][Const.HP] = {"D": inner_dim,
                                                                     "T": num_tokens,
                                                                     "k": k,
                                                                     "num_layers": num_layers_per_expert,
                                                                     "bias": bias,
                                                                     "dropout_p": dropout_p}
        self.token_experts = nn.ModuleList(
            [self.init_sub_model("PerTokenFFN") for _ in range(self.Ne)]
        )
        self.ln = torch.nn.LayerNorm((inner_dim,), eps=1e-7)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.router.weight)

    def forward(self, s: torch.Tensor):
        """
        Args:
            s: (bs, T, D)

        Returns:
            v: (bs, T, D)  -- routed mixture of experts output
        """

        B, T, D = s.size()
        # Router logits -> ReLU gates (no softmax, no top-k)
        gates = F.relu(self.router(s))
        float_mask = (gates > 0).float().detach()
        reg_loss = gates.sum(-1).sum(-1)
        sparsity = float_mask.sum() / (T * self.Ne)
        expert_outputs = torch.stack([exp(s) for exp in self.token_experts], dim=2)

        v = torch.einsum("btjd,btj->btd", expert_outputs, gates)

        return self.ln(s + v), (reg_loss, sparsity)


@ModelRegistry.register(req_subs={"TokenMxing", "SparseMoE"})
class RankMixerBlock(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        dim_per_token = model_cfg[Const.HP].get("dim_per_token")
        all_dim = model_cfg[Const.HP].get("all_dim")
        num_experts = model_cfg[Const.HP].get("num_experts")
        inner_dim = model_cfg[Const.HP].get("inner_dim")
        ffn_layers = model_cfg[Const.HP].get("ffn_layers")
        T = model_cfg[Const.HP].get("T")
        k = model_cfg[Const.HP].get("k")
        k_multiplier = model_cfg[Const.HP].get("k_multiplier")
        dropout_p = model_cfg[Const.HP].get("dropout_p")

        self.model_cfg[Const.SUB_MODELS]["TokenMxing"][Const.HP] = {"dim_per_token": dim_per_token,
                                                                    "x_dim": all_dim,
                                                                    "k": k,
                                                                    "k_multiplier": k_multiplier}
        self.tokenmixing = self.init_sub_model("TokenMxing")

        self.model_cfg[Const.SUB_MODELS]["SparseMoE"][Const.HP] = {"num_experts_per_token": num_experts,
                                                                   "inner_dim": inner_dim,
                                                                   "num_tokens": T,
                                                                   "k": k,
                                                                   "num_layers_per_expert": ffn_layers,
                                                                   "bias": True,
                                                                   "dropout_p": dropout_p
                                                                   }

        self.moe = self.init_sub_model("SparseMoE")

    def forward(self, x: torch.Tensor, loss_old: torch.Tensor, sparsity_old: torch.Tensor):
        x = self.tokenmixing(x)
        y, (loss, sparsity) = self.moe(x)
        return y, (loss + loss_old, sparsity + sparsity_old)


@ModelRegistry.register(req_subs={"RankMixingInput", "RankMixerBlock"})
class RankMixer(BaseModel):
    """
    某节跳动的RankMixer模型，根据论文复现。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)
        feat_conf = common_hp["feature_conf"]
        user_dim, item_dim = compute_user_item_feature_dims(feat_conf, model_conf)
        seq_feature_conf = feat_conf.get("seq_feature_columns")
        seq_lens = torch.tensor([
            max(subcfg["length"] for subcfg in feature_dict.values())
            for feature_dict in seq_feature_conf.values()
        ])
        self.register_buffer("seq_lens", seq_lens)
        num_seqs = len(seq_lens)
        seq_dim = num_seqs * self._embedding_dim
        all_dim = user_dim + item_dim + seq_dim
        dim_per_token = model_cfg[Const.HP].get("dim_per_token", 428)
        # 如果不能被整除，找到最近的一个可以被整除的数，并且修改为新的dim_per_token
        if all_dim % dim_per_token != 0:
            new_dim_per_token = self.adjust_dim_fast(all_dim, dim_per_token)
            dim_per_token = new_dim_per_token
            model_cfg[Const.HP]["dim_per_token"] = dim_per_token

        num_experts = model_cfg[Const.HP].get("num_experts", 10)
        k = model_cfg[Const.HP].get("k", 4)
        k_multiplier = model_cfg[Const.HP].get("k_multiplier", 1)
        ffn_layers = model_cfg[Const.HP].get("ffn_layers", 2)
        n_layers = model_cfg[Const.HP].get("n_layers", 2)
        dropout_p = model_cfg[Const.HP].get("dropout", 0.05)
        T = all_dim // dim_per_token
        inner_dim = int(T * k * k_multiplier)

        self.model_cfg[Const.SUB_MODELS]["RankMixingInput"][Const.HP] = {"dim_per_token": dim_per_token,
                                                                         "x_dim": all_dim,
                                                                         "k": k,
                                                                         "k_multiplier": k_multiplier}
        self.input_model = self.init_sub_model("RankMixingInput")

        self.model_cfg[Const.SUB_MODELS]["RankMixerBlock"][Const.HP] = {"dim_per_token": dim_per_token,
                                                                        "all_dim": all_dim,
                                                                        "num_experts": num_experts,
                                                                        "inner_dim": inner_dim,
                                                                        "ffn_layers": ffn_layers,
                                                                        "T": T,
                                                                        "k": k,
                                                                        "k_multiplier": k_multiplier,
                                                                        "dropout_p": dropout_p
                                                                        }
        self.rm_blocks = nn.Sequential(*[
            self.init_sub_model("RankMixerBlock")
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(inner_dim, self._embedding_dim, bias=False)

    def adjust_dim_fast(self, all_dim, dim_per_token):
        divisors = [d for d in range(1, all_dim + 1) if all_dim % d == 0]
        return min(divisors, key=lambda x: abs(x - dim_per_token))

    def mean_pool_subseqs(self, seq_feature_embs: torch.Tensor,
                          past_lengths: torch.Tensor,
                          seq_lens):
        """
        seq_feature_embs: (B, L, D)  concatenation of N sub-sequences of lengths given by seq_lens
        past_lengths:     (B, N)     valid length for each sub-sequence per batch
        seq_lens:         list[int] or 1D tensor of length N, sum(seq_lens) == L
        returns:          (B, N, D)  mean-pooled embeddings per sub-sequence
        """
        B, L, D = seq_feature_embs.shape
        device = seq_feature_embs.device
        dtype = seq_feature_embs.dtype

        N = seq_lens.numel()

        # For each position in [0..L-1], which sub-seq does it belong to?
        seq_ids = torch.arange(N, device=device).repeat_interleave(seq_lens)  # (L,)
        # One-hot assignment of positions to sub-seqs -> A: (N, L)
        A = F.one_hot(seq_ids, num_classes=N).T.to(dtype)  # (N, L)

        # Local index within its sub-seq for each position (0..L_i-1), then expand into (N, L)
        offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                             seq_lens.cumsum(0)[:-1]])  # (N,)
        offsets_per_pos = offsets[seq_ids]  # (L,)
        local_pos_per_pos = torch.arange(L, device=device) - offsets_per_pos  # (L,)
        P = A * local_pos_per_pos.unsqueeze(0)  # (N, L)

        # Build (B, N, L) mask: pick tokens that (a) belong to sub-seq i and (b) index < past_lengths[b, i]
        valid_len = past_lengths.to(torch.long).unsqueeze(-1)  # (B, N, 1)
        mask = (P.unsqueeze(0) < valid_len) & (A.unsqueeze(0).bool())  # (B, N, L)
        w = mask.to(dtype)  # (B, N, L)

        # Sum and divide by counts
        sums = torch.einsum('bnl,bld->bnd', w, seq_feature_embs)  # (B, N, D)
        counts = past_lengths.clamp(min=1).unsqueeze(-1).to(dtype)  # (B, N, 1)
        means = sums / counts  # (B, N, D)
        return means

    def forward(self,
                past_ids,
                num_rerank,
                model_inputs,
                user_feature_embs_original,
                item_feature_embs_original,
                seq_feature_embs):
        past_lengths = model_inputs['past_lengths']
        results = []
        B = seq_feature_embs.size(0)
        user_feature_embs_original = user_feature_embs_original.unsqueeze(1).expand(-1, num_rerank, -1)
        # 形状（B, N, D_u + D_i）
        x = torch.cat([user_feature_embs_original, item_feature_embs_original], dim=-1)
        seq_feature_embs = self.mean_pool_subseqs(seq_feature_embs, past_lengths,
                                                  self.seq_lens).view(B, 1, -1).expand(-1, num_rerank, -1)
        # 形状（B, N, D_u + D_i + D_s）
        x = torch.cat([x, seq_feature_embs], dim=-1)
        B, N, _ = x.size()
        x_batch = x.view(B * N, -1)
        rm_in = self.input_model(x_batch)
        l1_loss = 0.0
        sparsity = 0.0
        num_block = len(self.rm_blocks)
        for i in range(num_block):
            block_i = self.rm_blocks[i]
            rm_in, (l1_loss, sparsity) = block_i(rm_in, l1_loss, sparsity)
        rm_out = rm_in.mean(dim=1)
        l1_loss = l1_loss / len(self.rm_blocks)
        sparsity = sparsity / len(self.rm_blocks)
        output = self.output_proj(rm_out)
        y = output.view(B, N, -1)

        return {"deep_outputs": y, "deep_loss": (l1_loss, sparsity)}


@ModelRegistry.register()
class NoDLRM(DLRModule):
    """
    用户-物品-评分输入特征预处理模块, 用于处理用户、物品和评分的特征。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

    def forward(self,
                past_ids,
                num_rerank,
                model_inputs,
                user_feature_embs_original,
                item_feature_embs_original,
                seq_feature_embs):
        B = seq_feature_embs.size(0)
        N = num_rerank
        D = self._embedding_dim
        device = seq_feature_embs.device
        return {"deep_outputs": torch.zeros((B, N, D), device=device), "deep_loss": 0.0}
