import functools

import torch
import torch_npu

try:
    from torch_npu import gather_for_rank1 as index_select_func
except ImportError:
    from torch import index_select

    index_select_func = functools.partial(index_select, dim=0)


class IndexSelection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index):
        if x.dim() == 1:
            # gather_for_rank1 supports only 1-dim tensors
            result = index_select_func(x, index=index)
        else:
            result = torch.index_select(x, dim=0, index=index)
        ctx.save_for_backward(x, index)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, index = ctx.saved_tensors
        grad_x, grad_index = torch_npu.index_select_for_rank1_backward(grad_output, x, index)
        return grad_x, grad_index


def index_select(x, index):
    return IndexSelection.apply(x, index)
