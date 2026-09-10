import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# =========================
# 1. FiLM Generator (stable)
# =========================
class FiLMGenerator(nn.Module):
    def __init__(self, action_emb_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(action_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim * 2)
        )

        # stable init
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, action_embeddings: torch.Tensor):
        out = self.net(action_embeddings)
        scale, shift = out.split(out.size(-1) // 2, dim=-1)

        # stabilize scale around 1
        scale = 1.0 + torch.tanh(scale)
        return scale, shift


# =========================
# 2. Action-conditioned core (Fused FiLM + Gate)
# =========================
class ActionConditioningCore(nn.Module):
    def __init__(self, action_emb_dim: int, token_emb_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.film = FiLMGenerator(action_emb_dim, hidden_dim, token_emb_dim)

        self.gate = nn.Sequential(
            nn.Linear(action_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_emb_dim)
        )

        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x: torch.Tensor, action_emb: torch.Tensor):
        """
        x: [B, N, D]
        action_emb: [B, N, Da]
        """

        scale, shift = self.film(action_emb)        # [B, N, D]
        gate = torch.sigmoid(self.gate(action_emb)) # [B, N, D]

        # fused computation (NO intermediate x_film)
        return x + gate * ((scale - 1.0) * x + shift)


# =========================
# 3. Attention Bias (NPU-friendly)
# =========================
class AttentionBiasGenerator(nn.Module):
    """
    Generate a key-side attention bias from observed history action types.

    Candidate action types are labels in the current training data, so they must
    never be used to build an attention bias for candidate prediction.
    """

    def __init__(self, num_action_types: int, bias_scale: float = 0.1):
        super().__init__()

        self.bias = nn.Embedding(num_action_types + 2, 1, padding_idx=0)
        self.scale = bias_scale

        nn.init.zeros_(self.bias.weight)

    def forward(
        self,
        action_types: torch.Tensor,
        seq_len: int,
        has_user_token: bool = True,
        candidate_len: int = 0,
    ):
        """
        action_types: [B, H] observed history action ids.
        return: [B, 1, seq_len], broadcast on query rows and attention heads.
        """

        bias = self.bias(action_types).squeeze(-1) * self.scale  # [B, N]

        if has_user_token:
            pad = torch.zeros(bias.size(0), 1, device=bias.device, dtype=bias.dtype)
            bias = torch.cat([pad, bias], dim=1)

        if candidate_len > 0:
            cand_pad = torch.zeros(bias.size(0), candidate_len, device=bias.device, dtype=bias.dtype)
            bias = torch.cat([bias, cand_pad], dim=1)

        if bias.size(1) != seq_len:
            if bias.size(1) > seq_len:
                bias = bias[:, :seq_len]
            else:
                pad = torch.zeros(
                    bias.size(0), seq_len - bias.size(1), device=bias.device, dtype=bias.dtype
                )
                bias = torch.cat([bias, pad], dim=1)

        return bias.unsqueeze(1)  # [B, 1, N]


# =========================
# 4. Full Action Conditioning Module
# =========================
class ActionConditioningModule(nn.Module):
    """
    Production-ready version:
    - FiLM + Gate fused
    - Attention bias optimized (no NxN)
    """

    def __init__(
        self,
        action_emb_dim: int,
        token_emb_dim: int,
        num_action_types: int,
        use_attention_biasing: bool = True,
        hidden_dim: int = 128,
        bias_scale: float = 0.1,
        use_film: bool = True,
        use_gated_fusion: bool = True,
        film_hidden_dim: Optional[int] = None,
        gate_hidden_dim: Optional[int] = None,
        attention_bias_scale: Optional[float] = None,
    ):
        super().__init__()

        hidden_dim = film_hidden_dim or gate_hidden_dim or hidden_dim
        bias_scale = attention_bias_scale if attention_bias_scale is not None else bias_scale

        self.core = ActionConditioningCore(
            action_emb_dim=action_emb_dim,
            token_emb_dim=token_emb_dim,
            hidden_dim=hidden_dim
        )

        self.use_attention_biasing = use_attention_biasing

        if use_attention_biasing:
            self.att_bias = AttentionBiasGenerator(
                num_action_types=num_action_types,
                bias_scale=bias_scale
            )

    def forward(
        self,
        x: torch.Tensor,
        action_emb: Optional[torch.Tensor] = None,
        action_embeddings: Optional[torch.Tensor] = None,
        action_types: Optional[torch.Tensor] = None,
    ):
        """
        x: [B, N, D]
        action_emb: [B, N, Da]
        """

        if action_emb is None:
            action_emb = action_embeddings
        if action_emb is None:
            raise ValueError("action_emb or action_embeddings must be provided")

        return self.core(x, action_emb)

    def generate_attention_bias(
        self,
        action_types: torch.Tensor,
        seq_len: Optional[int] = None,
        has_user_token: bool = True,
        candidate_len: int = 0,
        seq_length: Optional[int] = None,
    ) -> Optional[torch.Tensor]:

        if not self.use_attention_biasing:
            return None

        if seq_len is None:
            seq_len = seq_length
        if seq_len is None:
            raise ValueError("seq_len or seq_length must be provided")

        return self.att_bias(action_types, seq_len, has_user_token, candidate_len)