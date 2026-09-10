"""
MixFormer 专用网络层模块。

从 MixFormer 参考项目迁移的核心网络层：
- HeadMixing: 解耦的用户/物品头部混合模块
- BatchedPerHeadGLUFFN: 逐头批处理 GLU+FFN
- BatchedPerHeadLinear: 逐头批处理线性层
- _apply_norm: 统一处理不同格式的张量归一化
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_norm(x: torch.Tensor, norm: nn.Module, num_heads: int, embedding_dim: int) -> torch.Tensor:
    """
    统一处理 jagged 和标准格式下的 per-head RMSNorm。
    - 2D  Jagged: (T, H*D)
    - 3D  标准:   (B, S, H*D)
    - 4D  标准:   (B, S, H, D)
    """
    original_shape = x.shape

    if x.dim() == 2:
        x = x.contiguous().view(original_shape[0], num_heads, embedding_dim)
    elif x.dim() == 4:
        pass
    elif x.dim() == 3:
        x = x.contiguous().view(*original_shape[:2], num_heads, embedding_dim)
    else:
        raise ValueError(f"Unsupported input dimension: {x.dim()}D, shape={original_shape}")

    x = norm(x)

    return x.contiguous().view(original_shape)


class HeadMixing(nn.Module):
    """
    Decoupled Head Mixing module for multi-candidate sequence scenarios.

    This module performs head mixing between user and item features without
    explicitly constructing mask matrices, using tensor slicing for efficiency.
    """

    def __init__(self, num_user_heads: int, num_item_heads: int, d_model: int):
        super(HeadMixing, self).__init__()
        self.num_user_heads = num_user_heads
        self.num_item_heads = num_item_heads
        self.num_total_heads = num_user_heads + num_item_heads

        if d_model % self.num_total_heads != 0:
            raise ValueError(
                f"d_model must be divisible by total head count, "
                f"got {d_model} and {self.num_total_heads}"
            )
        self.head_dim = d_model // self.num_total_heads

    def forward(
            self,
            user_feat: torch.Tensor,
            item_feat_seq: torch.Tensor,
            candidate_offsets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updated to handle actual shapes from flow:
            user_feat:  (B, 1, n_u, D)       — 4D (from MixFormerFeaturePreprocessor)
            item_feat_seq: (B, C, n_c * D)   — 3D (after MixFormerModule candidate reshape)
        D = _embedding_dim = total_heads * head_dim.

        Returns:
            user_mixed: (B, 1, n_u, D)
            item_mixed: (B, C, n_c, D)
        """
        B = user_feat.size(0)
        C = item_feat_seq.size(1)
        D = user_feat.size(-1)
        total_dim = self.num_total_heads * self.head_dim

        # =========================
        # 1. User-side mixing
        #   user_feat (B, 1, n_u, D)  →  view(B, n_u, total_heads, head_dim)
        # =========================
        # squeeze dim=1, then reshape 变量到 (B, n_u, total_heads, head_dim)
        # Note: D: 维度 total_dim = total_heads * head_dim
        u = user_feat.squeeze(1)  # (B, n_u, D) = (B, n_u, total_heads * head_dim)
        u_reshaped = u.view(B, self.num_user_heads, self.num_total_heads, self.head_dim)

        user_part = u_reshaped[:, :, :self.num_user_heads, :]  # (B, n_u, n_u, head_dim)

        zeros = torch.zeros(
            B, self.num_user_heads, self.num_item_heads, self.head_dim,
            device=user_feat.device, dtype=user_feat.dtype
        )  # (B, n_u, n_c, head_dim)

        user_mixed_3d = torch.cat([user_part, zeros], dim=2).reshape(
            B, self.num_user_heads, total_dim
        )  # (B, n_u, total_dim)

        # Reshape back to (B, 1, n_u, D) for residual compatibility with QueryMixer
        user_mixed = user_mixed_3d.view(B, 1, self.num_user_heads, total_dim)

        # =========================
        # 2. Align user feature to item dimension
        # =========================
        # Expand user feature (B, 1, n_u, D) → (B, C, n_u, D) → (B, C, n_u * D)
        user_feat_aligned = user_feat.expand(B, C, self.num_user_heads, D).reshape(B, C, -1)

        # =========================
        # 3. Item-side mixing
        #   user_feat_aligned: (B, C, n_u * D)  →  view(B, C, n_u, total_heads, head_dim)
        #   item_feat_seq:     (B, C, n_c * D)  →  view(B, C, n_c, total_heads, head_dim)
        # =========================
        user_feat_aligned = user_feat_aligned.view(
            B, C, self.num_user_heads, self.num_total_heads, self.head_dim
        )

        item_feat_reshaped = item_feat_seq.view(
            B, C, self.num_item_heads, self.num_total_heads, self.head_dim
        )

        combined_feat = torch.cat([user_feat_aligned, item_feat_reshaped], dim=2)

        # Extract item portion (index beyond num_user_heads in the combined head dim)
        item_mixed = combined_feat.transpose(2, 3)[:, :, self.num_user_heads:, :, :].contiguous().view(
            B, C, self.num_item_heads, total_dim
        )

        return user_mixed, item_mixed


class BatchedPerHeadGLUFFN(nn.Module):
    """
    Per-head GLUFFN with batched einsum computation.

    Parameters are stored per-head:
        w13_weight: [num_heads, 2 * hidden_dim, input_dim]
        w2_weight:  [num_heads, output_dim, hidden_dim]
    """

    def __init__(
            self,
            num_heads: int,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            multiple_of: int = 2,
            ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim

        hidden_dim = int(2 * hidden_dim / 3)

        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        self.hidden_dim = multiple_of * (
                (hidden_dim + multiple_of - 1) // multiple_of
        )

        self.w13_weight = nn.Parameter(
            torch.empty(
                num_heads,
                2 * self.hidden_dim,
                input_dim,
            )
        )

        self.w2_weight = nn.Parameter(
            torch.empty(
                num_heads,
                output_dim,
                self.hidden_dim,
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        for head in range(self.num_heads):
            nn.init.xavier_uniform_(self.w13_weight[head])
            nn.init.xavier_uniform_(self.w2_weight[head])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., num_heads, input_dim) or (N, num_heads * input_dim)

        Returns:
            (..., num_heads, output_dim) or (N, num_heads * output_dim)
        """
        flatten_output = x.dim() == 2

        if flatten_output:
            if x.size(-1) != self.num_heads * self.input_dim:
                raise ValueError(
                    "input dim mismatch: got %s, expected %s"
                    % (x.size(-1), self.num_heads * self.input_dim)
                )
            N = x.size(0)
            x = x.reshape(N, self.num_heads, self.input_dim)
            batch_shape = (N,)
        else:
            if x.size(-2) != self.num_heads or x.size(-1) != self.input_dim:
                raise ValueError(
                    "input shape mismatch: got %s, expected (..., %s, %s)"
                    % (tuple(x.shape), self.num_heads, self.input_dim)
                )
            batch_shape = x.shape[:-2]
            x = x.reshape(-1, self.num_heads, self.input_dim)

        w13_out = torch.einsum(
            "nhd,hod->nho",
            x,
            self.w13_weight,
        )

        w1_x, w3_x = w13_out.split(self.hidden_dim, dim=-1)
        hidden = F.silu(w1_x) * w3_x

        out = torch.einsum(
            "nhd,hod->nho",
            hidden,
            self.w2_weight,
        )

        if flatten_output:
            return out.reshape(
                batch_shape[0],
                self.num_heads * self.output_dim,
            )

        return out.reshape(
            *batch_shape,
            self.num_heads,
            self.output_dim,
        )


class BatchedPerHeadLinear(nn.Module):
    def __init__(self, num_heads: int, in_features: int, out_features: int):
        super().__init__()

        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(num_heads, out_features, in_features)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., num_heads, in_features)

        Returns:
            (..., num_heads, out_features)
        """
        if x.size(-2) != self.num_heads:
            raise ValueError(
                "head dim mismatch: got %s, expected %s"
                % (x.size(-2), self.num_heads)
            )

        if x.size(-1) != self.in_features:
            raise ValueError(
                "input dim mismatch: got %s, expected %s"
                % (x.size(-1), self.in_features)
            )

        original_shape = x.shape[:-2]
        x = x.reshape(-1, self.num_heads, self.in_features)

        out = torch.einsum(
            "nhd,hod->nho",
            x,
            self.weight,
        )

        return out.reshape(*original_shape, self.num_heads, self.out_features)