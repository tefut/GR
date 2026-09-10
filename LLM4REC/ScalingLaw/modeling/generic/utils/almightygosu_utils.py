import torch
import torch_npu

from utils.logging_utils import logging


class IndexSelection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index):
        result = torch.index_select(x, dim=0, index=index)
        ctx.save_for_backward(x, index)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, index = ctx.saved_tensors
        grad_x, grad_index = torch_npu.index_select_for_rank1_backward(grad_output, x, index)
        return grad_x, grad_index


class IndexSelectionV1(IndexSelection):
    @staticmethod
    def forward(ctx, x, index):
        if x.dim() == 1:
            # gather_for_rank1 supports only 1-dim tensors
            result = torch_npu.gather_for_rank1(x, index=index)
        else:
            result = torch.index_select(x, dim=0, index=index)
        ctx.save_for_backward(x, index)
        return result


class IndexSelectionV2(IndexSelection):
    @staticmethod
    def forward(ctx, x, index):
        if x.dim() == 1:
            # gather_for_rank1 supports only 1-dim tensors
            result = torch.ops.mxrec.gather_for_rank1(x, index=index)
        else:
            result = torch.index_select(x, dim=0, index=index)
        ctx.save_for_backward(x, index)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, index = ctx.saved_tensors
        grad_x, grad_index = torch.ops.mxrec.index_select_for_rank1_backward(grad_output, x, index)
        return grad_x, grad_index


if (
        hasattr(torch.ops, "mxrec")
        and hasattr(torch.ops.mxrec, "gather_for_rank1")
        and hasattr(torch.ops.mxrec, "index_select_for_rank1_backward")
):
    def index_select(x, index):
        return IndexSelectionV2.apply(x, index)
    logging.info("Using mxrec acceleration ops V2 for index_select.")
elif hasattr(torch_npu, "gather_for_rank1") and hasattr(torch_npu, "index_select_for_rank1_backward"):
    def index_select(x, index):
        return IndexSelectionV1.apply(x, index)
    logging.info("Using mxrec acceleration ops V1 for index_select.")
else:
    def index_select(x, index):
        return IndexSelection.apply(x, index)
    logging.info(
        "Mxrec acceleration ops cannot found, using torch.index_select instead."
    )
