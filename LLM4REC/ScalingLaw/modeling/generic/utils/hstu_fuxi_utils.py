import logging
import os
from typing import Any
from collections import namedtuple

import torch


class HstuFuxi(torch.autograd.Function):
    """For FUXI dense fusion op"""

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        q, k, v, rel_ts_bias, rel_pos_bias, mask, mask_type, n, qk_attn_denominator_value, layout, seq_offset = args

        attn_output = torch.ops.mxrec.hstu_fuxi(
            q, k, v, rel_ts_bias, rel_pos_bias, mask,
            mask_type, n, qk_attn_denominator_value, layout, seq_offset)

        ctx.save_for_backward(q, k, v, rel_ts_bias, rel_pos_bias)
        ctx.attn_mask = mask
        ctx.mask_type = mask_type
        ctx.max_seq_len = n
        ctx.silu_scale = qk_attn_denominator_value
        ctx.layout = layout
        ctx.seq_offset = seq_offset
        return attn_output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        q, k, v, rel_ts_bias, rel_pos_bias = ctx.saved_tensors

        grad_outputs = grad_outputs[0]
        headNum = q.shape[1]
        headDime = q.shape[2]
        grad_outputs = grad_outputs.reshape(-1, headNum, headDime)
        q_grad, k_grad, v_grad, pos_grad, ts_grad = torch.ops.mxrec.hstu_dense_backward_fuxi(
            grad_outputs, q, k, v, ctx.attn_mask, rel_pos_bias, rel_ts_bias, ctx.layout,
            ctx.mask_type, ctx.max_seq_len, ctx.silu_scale, ctx.seq_offset)
        Gradients = namedtuple(
            "Gradients",
            ["q_grad", "k_grad", "v_grad", "ts_grad", "pos_grad",
             "r1", "r2", "r3", "r4", "r5", "r6"]
        )
        return Gradients(
            q_grad, k_grad, v_grad, ts_grad, pos_grad,
            None, None, None, None, None, None
        )


def hstu_fuxi(q, k, v, rel_ts_bias, rel_pos_bias, attn_mask,
              mask_type, max_seq_len, silu_scale, layout, seq_offset=None):
    if layout == "jagged" and seq_offset is None:
        raise RuntimeError("seq_offset cannot be None while layout is jagged")
    if mask_type == 3 and attn_mask is None:
        raise RuntimeError("attn_mask cannot be None while mask_type is custom(3)")
    return HstuFuxi.apply(q, k, v, rel_ts_bias, rel_pos_bias, attn_mask, mask_type,
                          max_seq_len, silu_scale, layout, seq_offset)
