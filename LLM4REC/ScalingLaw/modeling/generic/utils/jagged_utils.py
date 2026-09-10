from typing import Any

import torch

from modeling import JAGGED_OPS_VERSION, JAGGED_OPS_VERSION_V1, JAGGED_OPS_VERSION_V2

PADDING_VALUE = 0.


class Dense_to_Jagged(torch.autograd.Function):
    """For Dense_to_Jagged op"""

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x, x_offsets, dense_length, jagged_length = args

        x_jagged = torch.ops.mxrec.dense_to_jagged(x, [x_offsets], jagged_length)[0]

        ctx.x_offsets = x_offsets
        ctx.dense_length = dense_length
        return x_jagged

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_dense = torch.ops.mxrec.jagged_to_padded_dense(grad_outputs[0], [ctx.x_offsets], ctx.dense_length, 0.0)

        return grad_dense, None, None, None


class Jagged_to_Dense(torch.autograd.Function):
    """For Jagged_to_Dense op"""

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x, x_offsets, dense_length, jagged_length = args

        ctx.x_offsets = x_offsets
        ctx.jagged_length = jagged_length
        x_dense = torch.ops.mxrec.jagged_to_padded_dense(x, [x_offsets], dense_length, 0.0)

        return x_dense

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_jagged = torch.ops.mxrec.jagged_to_padded_dense_backward(
            grad_outputs[0], [ctx.x_offsets], ctx.jagged_length)

        return grad_jagged, None, None, None


def dense_to_jagged(dense: torch.Tensor, offsets: torch.Tensor, dense_length: int, jagged_length: int):
    if JAGGED_OPS_VERSION == JAGGED_OPS_VERSION_V1:
        return torch.ops.fbgemm.dense_to_jagged(dense, [offsets])[0]
    elif JAGGED_OPS_VERSION == JAGGED_OPS_VERSION_V2:
        return Dense_to_Jagged.apply(dense, offsets, dense_length, jagged_length)
    else:
        raise RuntimeError("Unknown dense_to_jagged ops version")


def jagged_to_padded_dense(values: torch.Tensor, offsets: torch.Tensor, dense_length: int, jagged_length: int):
    if JAGGED_OPS_VERSION == JAGGED_OPS_VERSION_V1:
        return torch.ops.fbgemm.jagged_to_padded_dense(values, [offsets], [dense_length], PADDING_VALUE)
    elif JAGGED_OPS_VERSION == JAGGED_OPS_VERSION_V2:
        return Jagged_to_Dense.apply(values, offsets, dense_length, jagged_length)
    else:
        raise RuntimeError("Unknown jagged_to_padded_dense ops version")
