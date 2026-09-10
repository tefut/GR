from typing import Any

import torch
import collections

HstuOp = collections.namedtuple("HstuOp",
                                ["q", "k", "v", "attn_mask", "rab", "max_seq_len", "silu_scale", "layout",
                                 "seq_offset"])


class HstuDense(torch.autograd.Function):
    """For hstu dense fusion op"""

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        q, k, v, attn_mask, rab, mask_type, max_seq_len, silu_scale, layout, seq_offset = args
        output = torch.ops.mxrec.hstu_dense(
            q, k, v, attn_mask, rab, mask_type, max_seq_len, silu_scale, layout, seq_offset
        )

        ctx.save_for_backward(q, k, v, rab)
        ctx.attn_mask = attn_mask
        ctx.mask_type = mask_type
        ctx.max_seq_len = max_seq_len
        ctx.silu_scale = silu_scale
        ctx.layout = layout
        ctx.seq_offset = seq_offset
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        q, k, v, rab = ctx.saved_tensors
        q_grad, k_grad, v_grad, rab_grad = torch.ops.mxrec.hstu_dense_backward(
            grad_outputs[0], q, k, v, ctx.attn_mask, rab, ctx.layout, ctx.mask_type, ctx.max_seq_len, ctx.silu_scale,
            ctx.seq_offset
        )
        if rab is None:
            rab_grad = None
        hstuop = HstuOp(q_grad, k_grad, v_grad, None, rab_grad, None, None, None, None, None)
        return hstuop


def hstu_dense(q, k, v, attn_mask, rab, mask_type, max_seq_len, silu_scale, layout, seq_offset=None):
    if layout == "jagged" and seq_offset is None:
        raise RuntimeError("seq_offset cannot be None while layout is jagged")
    if mask_type == 3 and attn_mask is None:
        raise RuntimeError("attn_mask cannot be None while mask_type is custom(3)")
    hstuop = HstuOp(q, k, v, attn_mask, rab, mask_type, max_seq_len, silu_scale, layout, seq_offset)
    return HstuDense.apply(hstuop)
