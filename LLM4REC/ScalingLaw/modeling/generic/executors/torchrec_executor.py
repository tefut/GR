import functools
from dataclasses import dataclass, field
from typing import cast, Iterator, TypeVar, Callable, Iterable, Optional, Any, Tuple, Dict, Type, Union, List

import copy
import torch
import torch_npu
import torch.autograd
import torch.distributed as dist
from functools import partial
import os
from torch import nn, optim
from torch.autograd.profiler import record_function
from torch.distributed.optim import _apply_optimizer_in_backward

import torchrec
from modeling import TORCHREC_VERSION, TORCHREC_VERSION_V11
from modeling.generic.executors.executor import Executor
from torch.distributed import rpc
from torchrec import KeyedJaggedTensor, JaggedTensor
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.model_parallel import (DistributedModelParallel, get_default_sharders)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology, ParameterConstraints
from torchrec.distributed.train_pipeline import _wait_for_batch
from torchrec.distributed.types import BoundsCheckMode
from torchrec.inference.state_dict_transform import state_dict_to_device
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.streamable import Pipelineable
from torchrec.distributed.train_pipeline.utils import PipelinedForward, TrainPipelineContext, _to_device
from torchrec.distributed.types import Awaitable, ShardingEnv
from torchrec.streamable import Multistreamable
from torchrec.distributed.embedding_types import KJTList
from modeling.generic.executors.unique_sharding_plan import get_default_unique_sharders

from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.types import (
    ShardingEnv,
    ModuleSharder,
    ShardingPlan,
    ShardedTensor,
    Shard,
    ShardedTensorMetadata,
)
import torch.distributed._shard.sharded_tensor.api as shard_api
from torch.distributed._shard.sharded_tensor import utils
import torch.distributed._shard.sharding_spec as shard_spec
from torch.distributed._shard.sharding_spec._internals import (
    check_tensor,
    validate_non_overlapping_shards_metadata,
)

import threading
import queue

In = TypeVar("In", bound=Pipelineable)
Out = TypeVar("Out")
T = TypeVar("T")


@dataclass
class Batch(Pipelineable):
    def __init__(self, payloads) -> None:
        self.payloads: Dict[str, torch.Tensor] = payloads

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        for name, feature in self.payloads.items():
            self.payloads[name] = feature.to(device, non_blocking=non_blocking)
        return self

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        for feature in self.payloads.values():
            feature.record_stream(stream)

    def pin_memory(self) -> "Batch":
        for feature in self.payloads.values():
            feature.pin_memory()
        return self

    def __getitem__(self, key):
        return self.payloads[key]


@dataclass
class UniqueEmbeddingTrainPipelineContext(TrainPipelineContext):
    compute_unique_feature_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    compute_and_output_result: Dict[str, Multistreamable] = field(default_factory=dict)
    i_batch = 0


@dataclass
class AsyncEmbeddingTrainPipelineContext(TrainPipelineContext):
    compute_and_output_result: Dict[str, Multistreamable] = field(default_factory=dict)
    compute_unique_feature_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    sparse_loss = None
    dense_loss = None
    detach_embedding = {}
    detach_embedding_grad = {}
    embedding = None


@dataclass
class UniqueEmbeddingTrainPipelineContext(TrainPipelineContext):
    compute_unique_feature_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    compute_and_output_result: Dict[str, Multistreamable] = field(default_factory=dict)


class UniqueAsyncEmbeddingPipelinedForward(PipelinedForward):

    def compute_feature_unique(self, stream: Optional[torch.cuda.streams.Stream]):
        with torch_npu.npu.stream(stream):
            if self._name not in self._context.input_dist_tensors_requests:
                raise RuntimeError(
                    "Invalid PipelinedForward usage, please do not directly call model.forward()"
                )
            request = self._context.input_dist_tensors_requests.pop(self._name)
            if not isinstance(request, Awaitable):
                raise TypeError(
                    f"Expected an Awaitable input request, but got {type(request).__name__}"
                )
            with record_function("## wait_sparse_data_dist ##"):
                data = request.wait()

            ctx = self._context.module_contexts[self._name]

            if hasattr(self._module, "unique_input_dist"):
                unique_data_awaitable = self._module.unique_input_dist(ctx, data)
                self._context.compute_unique_feature_requests[self._name] = unique_data_awaitable
            else:
                raise RuntimeError(
                    "TrainPipelineSparseDistAsyncEmbedding can't be used for module with no post_input method"
                )
            return

    def output(self, stream: Optional[torch.cuda.streams.Stream]):
        with torch_npu.npu.stream(stream):
            if self._name not in self._context.compute_unique_feature_requests:
                raise RuntimeError(
                    "Invalid PipelinedForward usage, please do not directly call model.forward()"
                )

            with record_function("## wait_unique ##"):
                # Finish waiting on the dist_stream,
                # in case some delayed stream scheduling happens during the wait() call.
                unique_data_list = []
                unique_inverse_list = []
                for request in (self._context.compute_unique_feature_requests.pop(self._name)):
                    current_unique_data, current_unique_inverse = request.wait()
                    unique_data_list.append(current_unique_data)
                    unique_inverse_list.append(current_unique_inverse)
                unique_data_list = KJTList(unique_data_list)
                unique_inverse_list = KJTList(unique_inverse_list)

            # Make sure that both result of input_dist and context
            # are properly transferred to the current stream.
            ctx = self._context.module_contexts[self._name]

            unique_embs = self._module.compute(ctx, unique_data_list)
            origin_emb_list = []
            for index, unique_emb in enumerate(unique_embs):
                origin_emb = torch.index_select(input=unique_emb, dim=0, index=unique_inverse_list[index])
                origin_emb_list.append(origin_emb)
            embedding = self._module.output_dist(ctx, origin_emb_list).wait()
            ctx.record_stream(stream)
            return embedding

    def __call__(self, *args, **kwargs) -> Awaitable:

        data = self._context.detach_embedding[self._name]
        ctx = self._context.module_contexts.pop(self._name)

        return data


class UniqueAsyncEmbeddingPipelinedForwardEval(PipelinedForward):

    def compute_feature_unique(self, stream: Optional[torch.cuda.streams.Stream]):
        with torch_npu.npu.stream(stream):
            request = self._context.input_dist_tensors_requests.pop(self._name)
            with record_function("## wait_sparse_data_dist ##"):
                data = request.wait()

            ctx = self._context.module_contexts[self._name]

            if hasattr(self._module, "unique_input_dist"):
                unique_data_awaitable = self._module.unique_input_dist(ctx, data)
                self._context.compute_unique_feature_requests[self._name] = unique_data_awaitable
            else:
                raise RuntimeError(
                    "TrainPipelineSparseDistAsyncEmbedding can't be used for module with no post_input method"
                )
            return

    def output(self, stream: Optional[torch.cuda.streams.Stream]):
        with torch_npu.npu.stream(stream):
            with record_function("## wait_unique ##"):
                # Finish waiting on the dist_stream,
                # in case some delayed stream scheduling happens during the wait() call.
                unique_data_list = []
                unique_inverse_list = []
                for request in (self._context.compute_unique_feature_requests.pop(self._name)):
                    current_unique_data, current_unique_inverse = request.wait()
                    unique_data_list.append(current_unique_data)
                    unique_inverse_list.append(current_unique_inverse)
                unique_data_list = KJTList(unique_data_list)
                unique_inverse_list = KJTList(unique_inverse_list)

            # Make sure that both result of input_dist and context
            # are properly transferred to the current stream.
            ctx = self._context.module_contexts[self._name]

            unique_embs = self._module.compute(ctx, unique_data_list)
            origin_emb_list = []
            for index, unique_emb in enumerate(unique_embs):
                origin_emb = torch.index_select(input=unique_emb, dim=0, index=unique_inverse_list[index])
                origin_emb_list.append(origin_emb)
            self._context.compute_and_output_result[self._name] = self._module.output_dist(ctx, origin_emb_list)

            ctx.record_stream(stream)

    def __call__(self, *args, **kwargs) -> Awaitable:

        data = self._context.compute_and_output_result.pop(self._name).wait()
        ctx = self._context.module_contexts.pop(self._name)

        return data


class inputUniquePipelinedForward(PipelinedForward):

    def compute_feature_unique(self, stream: Optional[torch.cuda.streams.Stream] = None):
        with torch_npu.npu.stream(self._stream):
            request = self._context.input_dist_tensors_requests.pop(self._name)
            with record_function("## wait_sparse_data_dist ##"):
                data = request.wait()

            ctx = self._context.module_contexts[self._name]

            if hasattr(self._module, "unique_input_dist"):
                unique_data_awaitable = self._module.unique_input_dist(ctx, data)
                self._context.compute_unique_feature_requests[self._name] = unique_data_awaitable
            else:
                raise RuntimeError(
                    "UniqueTrainPipelineSparseDistWithReturnBatch can't be used for module with no post_input method"
                )
            return

    def __call__(self, *args, **kwargs) -> Awaitable:
        with record_function("## wait_unique ##"):
            # Finish waiting on the dist_stream,
            # in case some delayed stream scheduling happens during the wait() call.
            with torch.get_device_module(self._device).stream(self._stream):
                unique_data_list = []
                unique_inverse_list = []
                for request in (self._context.compute_unique_feature_requests.pop(self._name)):
                    current_unique_data, current_unique_inverse = request.wait()
                    unique_data_list.append(current_unique_data)
                    unique_inverse_list.append(current_unique_inverse)
                unique_data_list = KJTList(unique_data_list)
                unique_inverse_list = KJTList(unique_inverse_list)

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        ctx = self._context.module_contexts.pop(self._name)

        if self._stream is not None:
            torch.get_device_module(self._device).current_stream().wait_stream(
                self._stream
            )
            cur_stream = torch.get_device_module(self._device).current_stream()

            unique_data_list.record_stream(cur_stream)
            unique_inverse_list.record_stream(cur_stream)
            ctx.record_stream(cur_stream)

        unique_embs = self._module.compute(ctx, unique_data_list)
        origin_emb_list = []
        for index, unique_emb in enumerate(unique_embs):
            origin_emb = torch.index_select(input=unique_emb, dim=0, index=unique_inverse_list[index])
            origin_emb_list.append(origin_emb)

        return self._module.output_dist(ctx, origin_emb_list)


class TrainPipelineSparseDistWithReturnBatch(TrainPipelineSparseDist):
    """Base"""

    def __init__(self, *args, use_loss_weighted_grad=False, node_num=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_loss_weighted_grad = use_loss_weighted_grad
        self.node_num = node_num

    @property
    def training(self):
        return self._model.training

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def reset(self):
        raise NotImplementedError("'reset' func is abstract")


class NewTrainPipelineSparseDistAsyncEmbedding(TrainPipelineSparseDistWithReturnBatch):

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 execute_all_batches: bool = True,
                 apply_jit: bool = False,
                 context_type: Type[TrainPipelineContext] = TrainPipelineContext,
                 # keep for backward compatibility
                 pipeline_postproc: bool = False,
                 custom_model_fwd: Optional[
                     Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
                 ] = None,
                 use_loss_weighted_grad=False,
                 node_num=1):
        super().__init__(model=model,
                         optimizer=optimizer,
                         device=device,
                         execute_all_batches=execute_all_batches,
                         apply_jit=apply_jit,
                         context_type=context_type,
                         pipeline_postproc=pipeline_postproc,
                         custom_model_fwd=custom_model_fwd)
        self._embedding_stream = (
            (torch.get_device_module(device).Stream(priority=-1))
            if device.type in ["cuda", "mtia", "npu"]
            else None
        )
        self._dense_stream = (
            (torch.get_device_module(device).Stream(priority=-1))
            if device.type in ["cuda", "mtia", "npu"]
            else None
        )
        # 跨机场景下获取对应卡的通信组
        self.pair_group = self.get_pair_rank_group(self.node_num)
        self.embedding_weights = []
        # 获取当前卡的embedding权重并排序
        self.get_embedding_weights()
        self._context_type: Type[TrainPipelineContext] = AsyncEmbeddingTrainPipelineContext
        self._context: TrainPipelineContext = AsyncEmbeddingTrainPipelineContext(version=0)
        self.backward_sync_event = torch.npu.Event(enable_timing=False)

    def fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        # pipeline is already filled
        if len(self.batches) >= 4:
            return
        # executes last batch in pipeline
        if self.batches and self._execute_all_batches:
            return

        # batch i
        if not self.enqueue_batch(dataloader_iter):
            return

        self._init_pipelined_modules(
            self.batches[0],
            self.contexts[0],
            UniqueAsyncEmbeddingPipelinedForward,
        )  # start_sparse_data_dist
        # data_dist_stream中等待第一次all2all结束，并下发第二次all2all
        self.wait_sparse_data_dist(self.contexts[0])

        for module in self._pipelined_modules:
            self._set_module_context(self.contexts[0])
            module.forward.compute_feature_unique(self._data_dist_stream)
        with torch_npu.npu.stream(self._data_dist_stream):
            self.get_embedding(self.contexts[0], self.batches[0], self._data_dist_stream)
        # 等待data_dist_stream中第二次all2all结束
        _wait_for_batch(cast(In, self.batches[0]), self._data_dist_stream)
        with torch_npu.npu.stream(self._dense_stream):
            torch.get_device_module(self._device).current_stream().wait_stream(
                self._data_dist_stream
            )
            self.get_detach_embedding(self.contexts[0])
            self.contexts[0].dense_loss = self._model(self.batches[0].payloads)
            if self._model.training:
                self.contexts[0].dense_loss.sum().backward()
            self.store_embedding_grad(self.contexts[0])
            self.backward_sync_event.record(stream=self._dense_stream)

        # batch i+1
        if not self.enqueue_batch(dataloader_iter):
            return
        self.start_sparse_data_dist(self.batches[1], self.contexts[1])
        self.wait_sparse_data_dist(self.contexts[1])
        for module in self._pipelined_modules:
            self._set_module_context(self.contexts[1])
            module.forward.compute_feature_unique(self._data_dist_stream)
        with torch_npu.npu.stream(self._data_dist_stream):
            self.get_embedding(self.contexts[1], self.batches[1], self._data_dist_stream)
        _wait_for_batch(cast(In, self.batches[1]), self._data_dist_stream)
        with torch_npu.npu.stream(self._dense_stream):
            torch.get_device_module(self._device).current_stream().wait_stream(
                self._data_dist_stream
            )
            self.get_detach_embedding(self.contexts[1])

        # batch i+2
        if not self.enqueue_batch(dataloader_iter):
            return
        self.start_sparse_data_dist(self.batches[2], self.contexts[2])
        self.wait_sparse_data_dist(self.contexts[2])
        _wait_for_batch(cast(In, self.batches[2]), self._data_dist_stream)
        for module in self._pipelined_modules:
            self._set_module_context(self.contexts[2])
            module.forward.compute_feature_unique(self._data_dist_stream)

        # batch i+3
        if not self.enqueue_batch(dataloader_iter):
            return

    def progress(self, data_iter: Iterator):
        if self._model.training:
            return self.train_progress(data_iter)
        else:
            return self.eval_progress(data_iter)

    def train_progress(self, data_iter: Iterator):
        if not self._model_attached:
            self.attach(self._model)

        self.fill_pipeline(data_iter)
        if len(self.batches) < 2:
            raise StopIteration

        self._set_module_context(self.contexts[0])

        # batch i+1
        # forward
        curr_batch = self.batches[1].payloads
        with record_function("## forward ##"):
            with torch_npu.npu.stream(self._dense_stream):
                self._set_module_context(self.contexts[1])
                self.contexts[1].dense_loss = self._model(self.batches[1].payloads)

            # batch 0 sparse backward
            if self._model.training:
                with record_function("## sparse_backward ##"):
                    with torch_npu.npu.stream(self._data_dist_stream):
                        self.sparse_backward(self.contexts[0], self._data_dist_stream)

            if self._model.training:
                with record_function("## zero_grad ##"):
                    self._optimizer.zero_grad()

            # batch i+3
            if len(self.batches) >= 4:
                self.start_sparse_data_dist(self.batches[3], self.contexts[3])

            # batch i+2
            if len(self.batches) >= 3:
                with record_function("## sparse forward ##"):
                    with torch_npu.npu.stream(self._data_dist_stream):
                        self.get_embedding(self.contexts[2], self.batches[2], self._data_dist_stream)
                    self._set_module_context(self.contexts[1])

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                with torch_npu.npu.stream(self._dense_stream):
                    output = torch.sum(self.contexts[1].dense_loss, dim=0)
                    output.backward()
                    self.store_embedding_grad(self.contexts[1])

                    # 跨机all-reduce
                    if self.node_num > 1:
                        with record_function("## cross-machine All-Reduce ##"):
                            self.allreduce_embedding_weight(self.embedding_weights, self.pair_group)
                            self.allreduce_model_grads(self._model, self.pair_group)

            # batch i+3
            if len(self.batches) >= 4:
                self.wait_sparse_data_dist(self.contexts[3])
                for module in self._pipelined_modules:
                    self._set_module_context(self.contexts[3])
                    module.forward.compute_feature_unique(self._data_dist_stream)
                self._set_module_context(self.contexts[1])

        # batch i+2
        if len(self.batches) >= 3:
            _wait_for_batch(cast(In, self.batches[2]), self._data_dist_stream)
            self.get_detach_embedding(self.contexts[2])

        # batch i+4
        self.enqueue_batch(data_iter)

        if self._model.training:
            # update
            with record_function("## optimizer ##"):
                with torch_npu.npu.stream(self._dense_stream):
                    self._optimizer.step()
        self.dequeue_batch()
        return output, curr_batch

    def store_embedding_grad(self, contexts):
        for module in self._pipelined_modules:
            detach_embedding_grad = {}
            for key, _ in contexts.compute_and_output_result[module.forward._name].items():
                detach_embedding_grad[key] = contexts.detach_embedding[module.forward._name][key].values().grad.clone()
                detach_embedding_grad[key].record_stream(self._dense_stream)
            contexts.detach_embedding_grad[module.forward._name] = detach_embedding_grad

    def get_detach_embedding(self, contexts):
        for module in self._pipelined_modules:
            detach_embedding = {}
            detach_embedding_grad = {}
            detach_values = []
            for key, i_jagged_tensor in contexts.compute_and_output_result[module.forward._name].items():
                detach_value = i_jagged_tensor.values().detach().clone().requires_grad_(True)
                detach_embedding[key] = JaggedTensor(values=detach_value,
                                                     lengths=i_jagged_tensor.lengths().detach().clone())
                detach_value.retain_grad()

                # detach embedding
                detach_values.append(detach_value)
            contexts.detach_embedding[module.forward._name] = detach_embedding

    def get_embedding(self, contexts, bath, stream):
        for module in self._pipelined_modules:
            self._set_module_context(contexts)
            contexts.compute_and_output_result[module.forward._name] = module.forward.output(stream)
            for _, i_jagged_tensor in contexts.compute_and_output_result[module.forward._name].items():
                i_jagged_tensor.record_stream(stream)

    def sparse_backward(self, contexts, stream):

        self.backward_sync_event.wait(stream=stream)
        sparse_loss = 0
        for module in self._pipelined_modules:
            for key, i_jagged_tensor in contexts.compute_and_output_result[module.forward._name].items():
                sparse_loss += (i_jagged_tensor.values() *
                                contexts.detach_embedding_grad[module.forward._name][key]).sum()
        sparse_loss.record_stream(stream)
        sparse_loss.backward()

    def get_embedding_eval(self, contexts, bath, stream):
        torch.get_device_module(self._device).current_stream().wait_stream(
            stream
        )
        for module in self._pipelined_modules:
            self._set_module_context(contexts)
            contexts.detach_embedding[module.forward._name] = module.forward.output(stream)
            for _, i_jagged_tensor in contexts.detach_embedding[module.forward._name].items():
                i_jagged_tensor.record_stream(stream)

    def eval_progress(self, data_iter: Iterator):
        self.fill_pipeline_eval(data_iter)

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self._batch_i), self._data_dist_stream)
        self._start_sparse_data_dist(self._batch_ip1)

        self._batch_ip2 = self._copy_batch_to_gpu(data_iter)

        # forward
        curr_batch = self._batch_i.payloads
        with record_function("## forward ##"):
            output = self._model(self._batch_i.payloads)

        self._wait_sparse_data_dist()
        for module in self._pipelined_modules:
            self._set_module_context(self._context)
            module.forward.compute_feature_unique(self._data_dist_stream)
            module.forward.output(self._data_dist_stream)

        self._batch_i = self._batch_ip1
        self._batch_ip1 = self._batch_ip2

        return output, curr_batch

    def fill_pipeline_eval(self, dataloader_iter: Iterator[In]) -> None:
        # pipeline is already filled
        if self._batch_i and self._batch_ip1:
            return
        # executes last batch in pipeline
        if self._batch_i and self._execute_all_batches:
            return

        # batch 1
        self._batch_i = self._copy_batch_to_gpu(dataloader_iter)
        if self._batch_i is None:
            raise StopIteration

        self._init_pipelined_modules(self._batch_i, self._context, UniqueAsyncEmbeddingPipelinedForwardEval)
        self._start_sparse_data_dist(self._batch_i)
        self._wait_sparse_data_dist()
        for module in self._pipelined_modules:
            self._set_module_context(self._context)
            module.forward.compute_feature_unique(self._data_dist_stream)
            module.forward.output(self._data_dist_stream)
        # batch 2
        self._batch_ip1 = self._copy_batch_to_gpu(dataloader_iter)

    def clear_prefetched_batches(self):
        self._batch_i = None

    def reset(self):
        self._batch_i = None
        self._batch_ip1 = None
        self._batch_ip2 = None
        if self._optimizer is not None:
            self._optimizer.zero_grad(set_to_none=True)
            del self._optimizer
            self._optimizer = None

    def get_embedding_weights(self):
        embedding_weights = []
        for _, module in self._model.named_modules():
            if isinstance(module, ShardedEmbeddingCollection):
                for name, param in module.named_parameters():
                    if param.numel() == 0:
                        continue
                    embedding_weights.append((name, param))
        embedding_weights.sort(key=lambda x: x[0])
        self.embedding_weights = [param for _, param in embedding_weights]

    def get_pair_rank_group(self, node_num: int):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        npu_per_node = world_size // node_num  # 每个节点卡数

        # 设置配对的 group
        local_pair_group = None
        for local_card_idx in range(npu_per_node):
            group_ranks = [node * npu_per_node + local_card_idx for node in range(node_num)]
            group = dist.new_group(group_ranks)
            if rank in group_ranks:
                local_pair_group = group
        dist.barrier()
        return local_pair_group

    def flatten_grads(self, model):
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad.detach().contiguous().view(-1)
                grads.append(grad)
        flat_grad = torch.cat(grads)
        return flat_grad

    def unflatten_grads(self, model, flat_grad):
        pointer = 0
        for param in model.parameters():
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.copy_(flat_grad[pointer:pointer + numel].view_as(param.grad))
                pointer += numel

    def allreduce_embedding_weight(self, embedding_weights, group):
        group_size = dist.get_world_size(group=group)
        for weight in embedding_weights:
            dist.all_reduce(weight.data, op=dist.ReduceOp.SUM, group=group)
            weight.data /= group_size

    # 同步dense部分梯度
    def allreduce_model_grads(self, model, group):
        group_size = dist.get_world_size(group=group)
        flat_grad = self.flatten_grads(model)
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, group=group)
        flat_grad /= group_size
        self.unflatten_grads(model, flat_grad)


class UniqueTrainPipelineSparseDistWithReturnBatch(TrainPipelineSparseDistWithReturnBatch):

    def __init__(self, *args, use_loss_weighted_grad=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_loss_weighted_grad = use_loss_weighted_grad
        self._context_type: Type[TrainPipelineContext] = UniqueEmbeddingTrainPipelineContext
        self._context: TrainPipelineContext = UniqueEmbeddingTrainPipelineContext(version=0)
        # 跨机场景下获取对应卡的通信组
        self.pair_group = self.get_pair_rank_group(self.node_num)
        self.embedding_weights = []
        # 获取当前卡的embedding权重并排序
        self.get_embedding_weights()

    def _fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        """
        DEPRECATED: exists for backward compatibility
        """
        # pipeline is already filled
        if self._batch_i and self._batch_ip1:
            return
        # executes last batch in pipeline
        if self._batch_i and self._execute_all_batches:
            return

        # batch 1
        self._batch_i = self._copy_batch_to_gpu(dataloader_iter)
        if self._batch_i is None:
            raise StopIteration

        self._init_pipelined_modules(self._batch_i, self._context, inputUniquePipelinedForward)
        self._start_sparse_data_dist(self._batch_i)
        self._wait_sparse_data_dist()
        for module in self._pipelined_modules:
            self._set_module_context(self._context)
            module.forward.compute_feature_unique()

        # batch 2
        self._batch_ip1 = self._copy_batch_to_gpu(dataloader_iter)

    def progress(self, data_iter: Iterator):
        self._fill_pipeline(data_iter)

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self._batch_i), self._data_dist_stream)
        self._start_sparse_data_dist(self._batch_ip1)

        self._batch_ip2 = self._copy_batch_to_gpu(data_iter)

        # forward
        curr_batch = self._batch_i.payloads
        with record_function("## forward ##"):
            output = cast(torch.Tensor, self._model(self._batch_i.payloads))

        self._wait_sparse_data_dist()

        for module in self._pipelined_modules:
            self._set_module_context(self._context)
            module.forward.compute_feature_unique()

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                output = torch.sum(output, dim=0)
                if self.use_loss_weighted_grad:
                    output.backward(output)
                else:
                    output.backward()
            # 跨机all-reduce
            if self.node_num > 1:
                with record_function("## cross-machine All-Reduce ##"):
                    self.allreduce_embedding_weight(self.embedding_weights, self.pair_group)
                    self.allreduce_model_grads(self._model, self.pair_group)

        if self._model.training:
            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        self._batch_i = self._batch_ip1
        self._batch_ip1 = self._batch_ip2

        return output, curr_batch

    def reset(self):
        self._batch_i = None
        self._batch_ip1 = None
        self._batch_ip2 = None
        if self._optimizer is not None:
            self._optimizer.zero_grad(set_to_none=True)
            del self._optimizer
            self._optimizer = None

    def clear_prefetched_batches(self):
        self._batch_i = None
        self._batch_ip1 = None

    def copy_batch_to_gpu(
            self,
            dataloader_iter: Iterator[In],
    ) -> Tuple[Optional[In], Optional[TrainPipelineContext]]:
        """
        Retrieves batch from dataloader and moves it to the provided device.

        Raises:
            StopIteration: if the dataloader iterator is exhausted; unless
                `self._execute_all_batches=True`, then returns None.
        """
        context = self._create_context()
        with record_function(f"## copy_batch_to_gpu {self._next_index} ##"):
            with self._stream_context(self._memcpy_stream):
                batch = self._next_batch(dataloader_iter)

                if not self._execute_all_batches:
                    raise StopIteration
                return batch, context

    def get_embedding_weights(self):
        embedding_weights = []
        for _, module in self._model.named_modules():
            if isinstance(module, ShardedEmbeddingCollection):
                for name, param in module.named_parameters():
                    if param.numel() == 0:
                        continue
                    embedding_weights.append((name, param))
        embedding_weights.sort(key=lambda x: x[0])
        self.embedding_weights = [param for _, param in embedding_weights]

    def get_pair_rank_group(self, node_num: int):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        npu_per_node = world_size // node_num  # 每个节点卡数

        # 设置配对的 group
        local_pair_group = None
        for local_card_idx in range(npu_per_node):
            group_ranks = [node * npu_per_node + local_card_idx for node in range(node_num)]
            group = dist.new_group(group_ranks)
            if rank in group_ranks:
                local_pair_group = group
        dist.barrier()
        return local_pair_group

    def flatten_grads(self, model):
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad.detach().contiguous().view(-1)
                grads.append(grad)
        flat_grad = torch.cat(grads)
        return flat_grad

    def unflatten_grads(self, model, flat_grad):
        pointer = 0
        for param in model.parameters():
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.copy_(flat_grad[pointer:pointer + numel].view_as(param.grad))
                pointer += numel

    def allreduce_embedding_weight(self, embedding_weights, group):
        group_size = dist.get_world_size(group=group)
        for weight in embedding_weights:
            dist.all_reduce(weight.data, op=dist.ReduceOp.SUM, group=group)
            weight.data /= group_size

    # 同步dense部分梯度
    def allreduce_model_grads(self, model, group):
        group_size = dist.get_world_size(group=group)
        flat_grad = self.flatten_grads(model)
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, group=group)
        flat_grad /= group_size
        self.unflatten_grads(model, flat_grad)


class TrainPipelineSparseDistWithReturnBatchV11(TrainPipelineSparseDistWithReturnBatch):
    """For torchrec 1.1"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 跨机场景下获取对应卡的通信组
        self.pair_group = self.get_pair_rank_group(self.node_num)
        self.embedding_weights = []
        # 获取当前卡的embedding权重并排序
        self.get_embedding_weights()

    def progress(self, data_iter: Iterator):
        if not self._model_attached:
            self.attach(self._model)

        self.fill_pipeline(data_iter)
        if not self.batches:
            raise StopIteration

        self._set_module_context(self.contexts[0])

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self.batches[0]), self._data_dist_stream)

        if len(self.batches) >= 2:
            self.start_sparse_data_dist(self.batches[1], self.contexts[1])

        # batch i+2
        self.enqueue_batch(data_iter)

        # forward
        payloads = self.batches[0].payloads
        with record_function("## forward ##"):
            output = cast(torch.Tensor, self._model(payloads))

        if len(self.batches) >= 2:
            self.wait_sparse_data_dist(self.contexts[1])

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                output = torch.sum(output, dim=0)
                if self.use_loss_weighted_grad:
                    output.backward(output)
                else:
                    output.backward()

            # 跨机all-reduce
            if self.node_num > 1:
                with record_function("## cross-machine All-Reduce ##"):
                    self.allreduce_embedding_weight(self.embedding_weights, self.pair_group)
                    self.allreduce_model_grads(self._model, self.pair_group)

            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        self.dequeue_batch()
        return output, payloads

    def reset(self):
        self.batches.clear()
        self.contexts.clear()
        if self._optimizer is not None:
            self._optimizer.zero_grad(set_to_none=True)
            del self._optimizer
            self._optimizer = None
        self.detach()

    def get_embedding_weights(self):
        embedding_weights = []
        for _, module in self._model.named_modules():
            if isinstance(module, ShardedEmbeddingCollection):
                for name, param in module.named_parameters():
                    if param.numel() == 0:
                        continue
                    embedding_weights.append((name, param))
        embedding_weights.sort(key=lambda x: x[0])
        self.embedding_weights = [param for _, param in embedding_weights]

    def get_pair_rank_group(self, node_num: int):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        npu_per_node = world_size // node_num  # 每个节点卡数

        # 设置配对的 group
        local_pair_group = None
        for local_card_idx in range(npu_per_node):
            group_ranks = [node * npu_per_node + local_card_idx for node in range(node_num)]
            group = dist.new_group(group_ranks)
            if rank in group_ranks:
                local_pair_group = group
        dist.barrier()
        return local_pair_group

    def flatten_grads(self, model):
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad.detach().contiguous().view(-1)
                grads.append(grad)
        flat_grad = torch.cat(grads)
        return flat_grad

    def unflatten_grads(self, model, flat_grad):
        pointer = 0
        for param in model.parameters():
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.copy_(flat_grad[pointer:pointer + numel].view_as(param.grad))
                pointer += numel

    def allreduce_embedding_weight(self, embedding_weights, group):
        group_size = dist.get_world_size(group=group)
        for weight in embedding_weights:
            dist.all_reduce(weight.data, op=dist.ReduceOp.SUM, group=group)
            weight.data /= group_size

    # 同步dense部分梯度
    def allreduce_model_grads(self, model, group):
        group_size = dist.get_world_size(group=group)
        flat_grad = self.flatten_grads(model)
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, group=group)
        flat_grad /= group_size
        self.unflatten_grads(model, flat_grad)


class TrainPipelineSparseDistWithReturnBatchV05(TrainPipelineSparseDistWithReturnBatch):
    """For torchrec 0.5"""

    def progress(self, data_iter: Iterator):
        self._fill_pipeline(data_iter)

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self._batch_i), self._data_dist_stream)
        self._start_sparse_data_dist(self._batch_ip1)

        self._batch_ip2 = self._copy_batch_to_gpu(data_iter)

        # forward
        payloads = self._batch_i.payloads
        with record_function("## forward ##"):
            output = cast(torch.Tensor, self._model(payloads))

        self._wait_sparse_data_dist()

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                output = torch.sum(output, dim=0)
                if self.use_loss_weighted_grad:
                    output.backward(output)
                else:
                    output.backward()

            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        self._batch_i = self._batch_ip1
        self._batch_ip1 = self._batch_ip2

        return output, payloads

    def reset(self):
        self._batch_i = None
        self._batch_ip1 = None
        self._batch_ip2 = None
        if self._optimizer is not None:
            self._optimizer.zero_grad(set_to_none=True)
            del self._optimizer
            self._optimizer = None


class TorchrecExecutor(Executor):
    """Torchrec executor"""

    def __init__(self, model, config, world_size, node_num, device):
        super().__init__(config)
        self._model = model
        self._saved_model = copy.deepcopy(model).to_empty(device="cpu")
        self._save_pg = dist.new_group(backend="gloo")
        self._device = torch.device(device)
        self._batch_limit = config.get("batch_limit", 0)
        self._use_loss_weighted_grad = config.get("use_loss_weighted_grad", False)
        self._step_to_freeze_mlp = config.get("steps_to_freeze_mlp", False)
        self._mlp_frozen = False

        self._ec_feature_names = self._model.embedding_module.ec_feature_names
        self._ec_feature_counts = self._model.embedding_module.feature_max_number

        shard_api._parse_and_validate_remote_device = my_parse_and_validate_remote_device
        utils._parse_and_validate_remote_device = my_parse_and_validate_remote_device
        ShardedTensor._init_from_local_shards_and_global_metadata = classmethod(
            my_init_from_local_shards_and_global_metadata)
        ShardedTensor._init_from_local_shards = classmethod(my_init_from_local_shards)

        # _apply_optimizer_in_backward 必须在进行torchrec planer处理之前执行，否则会报错
        for ec_attr in self._ec_feature_names.keys():
            _apply_optimizer_in_backward(
                torch.optim.Adagrad,
                getattr(self._model.embedding_module, ec_attr).parameters(),
                {"lr": self._config["learning_rate"]},
            )

        constraints = {}
        feature_tables: Dict[str, torchrec.EmbeddingConfig] = self._model.embedding_module.feature_tables
        multi_feature_names = self._model.embedding_module.multi_feature_names
        for name, table in feature_tables.items():
            if name in multi_feature_names:
                # multit特征emb切分时不能使用row_wise，否则EC查表时会报错：
                # RuntimeError: cannot call get_autograd_meta() on undefined tensor
                constraints[name] = (
                    ParameterConstraints(sharding_types=["table_wise"], bounds_check_mode=BoundsCheckMode.NONE)
                    if table.num_embeddings < 20000 else
                    ParameterConstraints(sharding_types=["row_wise"], bounds_check_mode=BoundsCheckMode.NONE))
            else:
                constraints[name] = (
                    ParameterConstraints(sharding_types=["table_wise"], bounds_check_mode=BoundsCheckMode.NONE)
                    if table.num_embeddings < 20000 else
                    ParameterConstraints(sharding_types=["row_wise"], bounds_check_mode=BoundsCheckMode.NONE))

        npu_per_node = world_size // node_num
        rank = dist.get_rank()
        # 创建ShardingEnv和进程组
        env, self.intra_node_pg = self.create_sharding_env(world_size, rank, npu_per_node)

        sharders = get_default_unique_sharders()
        planner = EmbSardingPlanner(
            topology=Topology(
                local_world_size=npu_per_node,
                world_size=npu_per_node,
                compute_device="npu",
            ),
            constraints=constraints
        )

        plan = planner.collective_plan(
            self._model, rank, npu_per_node, sharders, self.intra_node_pg
        )

        self._dmp = self._dmp_producer(self._model, sharders, plan, env)

        if TORCHREC_VERSION == TORCHREC_VERSION_V11:
            pipeline_cls = UniqueTrainPipelineSparseDistWithReturnBatch
        else:
            pipeline_cls = TrainPipelineSparseDistWithReturnBatchV05

        def _create_pipeline_producer(dmp, optimizer):
            return pipeline_cls(
                dmp, optimizer, self._device,
                execute_all_batches=True,
                use_loss_weighted_grad=self._use_loss_weighted_grad,
                node_num=node_num
            )

        self._pipeline_producer = _create_pipeline_producer
        self._pipeline = self._pipeline_producer(self._dmp, None)

        self._data_iter = None  # lazy init

    def _dmp_producer(self, module, sharders, plan, env):
        return DistributedModelParallel(
            module=module,
            device=self._device,
            sharders=sharders,
            plan=plan,
            env=env,
            init_data_parallel=True
        )

    def _prepare_batch(self, payloads: Dict[str, Union[torch.Tensor, KeyedJaggedTensor]]):
        for ec_attr, feature_names in self._ec_feature_names.items():
            jt_dict = {}
            for feature_name in feature_names:
                feature_id = payloads[feature_name]
                batch_size, lengths = feature_id.size(0), feature_id.numel()

                feature_id = feature_id.reshape(-1).long()
                if feature_name in self._ec_feature_counts:
                    feature_max_number = self._ec_feature_counts[feature_name]
                    zero_mask = torch.eq(feature_id, 0)
                    zero_indices = torch.nonzero(zero_mask)
                    feature_id[zero_indices] = torch.randint(1, feature_max_number, (zero_indices.shape))

                jt = JaggedTensor(
                    values=feature_id.reshape(-1).long(),
                    lengths=torch.tensor([lengths // batch_size] * batch_size, dtype=torch.int64)
                )
                jt_dict[feature_name] = jt
            payloads[ec_attr] = KeyedJaggedTensor.from_jt_dict(jt_dict)
        batch = _to_device(Batch(payloads), self._device, non_blocking=True)
        return batch

    def evaluate_once(self, model_input: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """torchrec0.5 rewrite模型之后没有提供还原的方法，模型单独调用报错，因此调用progress方法兼容(1.1无该问题)"""
        self._pipeline.eval()
        input_iter = iter([self._prepare_batch(model_input).to(self._device)])
        output, _ = self._pipeline.progress(input_iter)
        # remove kjts setted during _prepare_batch
        self._del_dict_items(model_input, self._ec_feature_names.keys())
        return output

    def execute(self, batch_id) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        if self._pipeline.training and self._batch_limit and batch_id > self._batch_limit:
            self._pipeline.reset()
            raise StopIteration
        condition1 = self._pipeline.training and self._step_to_freeze_mlp
        condition2 = not self._mlp_frozen and batch_id > self._step_to_freeze_mlp
        if (
                condition1 and condition2
        ):
            self._pipeline.optimizer.zero_grad()
            for name, param in self._model.named_parameters():
                if "emb_mlp" in name or "feed_forward_" in name:
                    param.requires_grad = False
            self._dmp = self._dmp_producer(self._model)
            self._pipeline = self._pipeline_producer(self._dmp, self._pipeline.optimizer)
            self._mlp_frozen = True

        try:
            output, model_input = self._pipeline.progress(self._data_iter)
            # remove kjts setted during _prepare_batch
            self._del_dict_items(model_input, self._ec_feature_names.keys())
        except StopIteration as e:
            self._pipeline.reset()
            raise e
        return output, model_input

    def train(self, data_loader):
        self._pipeline.train()
        dense_optimizer = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(self._dmp.named_parameters())),
            self._get_optimizer,
        )
        self._pipeline.optimizer = CombinedOptimizer([self._dmp.fused_optimizer, dense_optimizer])
        self._data_iter = BackgroundGenerator(map(self._prepare_batch, iter(data_loader)), max_prefetch=3)

    def eval(self, data_loader):
        self._pipeline.eval()
        self._data_iter = map(self._prepare_batch, iter(data_loader))

    def create_sharding_env(self, world_size: int, rank: int, npu_per_node: int):
        node_num = world_size // npu_per_node

        intra_node_pg = None
        # 创建机器内进程组
        for node in range(node_num):
            group_ranks = list(range(node * npu_per_node, (node + 1) * npu_per_node))
            group = dist.new_group(group_ranks)
            if rank in group_ranks:
                intra_node_pg = group
        dist.barrier()

        # 创建ShardingEnv
        env = ShardingEnv(
            world_size=npu_per_node,
            rank=rank % npu_per_node,
            pg=intra_node_pg,
        )

        return env, intra_node_pg

    @property
    def model(self):
        return self._dmp

    def gather_state_dict(self, *args, **kwargs):
        # Copy sharded state_dict to CPU. 
        # 重要：由于模型权重分布式Shard时不跨机的优化，这里传入的pg为处理后的NPU pg，用于获取正确的分片信息；
        # 在state_dict_gather执行时替换回CPU pg
        cpu_state_dict = state_dict_to_device(
            self._dmp.state_dict(), pg=self.intra_node_pg, device=torch.device("cpu")
        )
        saved_model = self._saved_model.to_empty(device="cpu")
        self.state_dict_gather(cpu_state_dict, saved_model.state_dict())
        dist.barrier()
        return saved_model.state_dict()

    def state_dict_gather(
            self,
            src: Dict[str, Union[torch.Tensor, ShardedTensor]],
            dst: Dict[str, torch.Tensor],
    ) -> None:
        """
        
        Args:
            src (Dict[str, Union[torch.Tensor, ShardedTensor]]): source's state_dict for this rank
            dst (Dict[str, torch.Tensor]): destination's state_dict
        """
        for key, dst_tensor in dst.items():
            src_tensor = src[key]
            if isinstance(src_tensor, ShardedTensor):
                # 替换跨机优化的process_group为cpu的
                src_tensor._prepare_init(process_group=self._save_pg)  # noqa
                src_tensor.gather(out=dst_tensor if (dist.get_rank() == 0) else None)
            elif isinstance(src_tensor, torch.Tensor):
                dst_tensor.copy_(src_tensor)
            else:
                raise ValueError(f"Unsupported tensor {key} type {type(src_tensor)}")

    def reset(self):
        self._pipeline.reset()
        self._data_iter = None

    def close(self):
        self.reset()
        self._del_objs(self._pipeline, self._dmp, self._model)


class EmbSardingPlanner(EmbeddingShardingPlanner):
    def collective_plan(
            self,
            module: nn.Module,
            rank: int,
            npu_per_node: int,
            sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
            pg: Optional[dist.ProcessGroup] = dist.GroupMember.WORLD,
    ) -> ShardingPlan:
        """
        Call self.plan(...) on rank 0 and broadcast
        """
        if sharders is None:
            sharders = get_default_sharders()

        node_id = rank // npu_per_node
        first_rank = npu_per_node * node_id

        return my_invoke_on_rank_and_broadcast_result(
            pg,
            first_rank,
            self.plan,
            module,
            sharders,
        )


def my_invoke_on_rank_and_broadcast_result(
        pg: dist.ProcessGroup,
        rank: int,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
) -> T:
    if pg.rank() == 0:
        res = func(*args, **kwargs)
        object_list = [res]
    else:
        object_list = [None]
    if pg.size() > 1:
        dist.broadcast_object_list(object_list, rank, group=pg)
    return cast(T, object_list[0])


def my_parse_and_validate_remote_device(pg, remote_device):
    if remote_device is None:
        raise ValueError("remote device is None")

    worker_name = remote_device.worker_name()
    rank = remote_device.rank()
    device = remote_device.device()

    # Validate rank, skip validation if rank is not part of process group.
    if not dist._rank_not_in_group(pg):
        if rank is not None and (rank < 0 or rank >= dist.get_world_size(pg)):
            raise ValueError(f'Invalid rank: {rank}')

    if worker_name is not None:
        if not rpc._is_current_rpc_agent_set():
            raise RuntimeError(f'RPC framework needs to be initialized for using worker names: {worker_name}')

        workers = rpc._get_current_rpc_agent().get_worker_infos()
        for worker in workers:
            if worker.name == worker_name:
                return worker.id, device

        raise ValueError(f'Invalid worker name: {worker_name}')

    return rank, device


def my_init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls, local_shards: List[Shard], sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None, init_rrefs=False, sharding_spec=None,
) -> ShardedTensor:
    process_group = cls._normalize_pg(process_group)
    current_rank = dist.get_rank(process_group)

    shards_metadata = sharded_tensor_metadata.shards_metadata

    local_shard_metadatas = []

    for shard_metadata in shards_metadata:
        rank, local_device = my_parse_and_validate_remote_device(
            process_group, shard_metadata.placement
        )

        if current_rank == rank:
            local_shard_metadatas.append(shard_metadata)

    if len(local_shards) != len(local_shard_metadatas):
        raise RuntimeError(
            f"Number of local shards ({len(local_shards)}) does not match number of local "
            f"shards metadata in sharded_tensor_metadata ({len(local_shard_metadatas)}) "
            f"on rank ({current_rank}) "
        )

    shards_metadata = sharded_tensor_metadata.shards_metadata
    tensor_properties = sharded_tensor_metadata.tensor_properties

    if len(shards_metadata) == 0:
        raise ValueError("shards_metadata must not be empty!")

    if tensor_properties.layout != torch.strided:
        raise ValueError("Only torch.strided layout is currently supported")

    if sharding_spec is None:
        spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
    else:
        spec = sharding_spec

    sharded_tensor = ShardedTensor.__new__(
        ShardedTensor,
        spec,
        sharded_tensor_metadata.size,
        dtype=tensor_properties.dtype,
        layout=tensor_properties.layout,
        pin_memory=tensor_properties.pin_memory,
        requires_grad=tensor_properties.requires_grad,
    )

    def _raise_if_mismatch(expected, actual, prop_name, rank, is_property=False):
        tensor_property_or_metadata = (
            "tensor property" if is_property else "local ShardMetadata"
        )
        if expected != actual:
            raise ValueError(
                f"Local shards' tensor {prop_name} property is incompatible with "
                f"{tensor_property_or_metadata} on rank {rank}: "
                f"{tensor_property_or_metadata} {prop_name}={expected}, "
                f"local shard tensor {prop_name}={actual}."
            )

    for shard in local_shards:
        shard_meta = shard.metadata
        local_shard_tensor = shard.tensor
        placement = shard_meta.placement
        rank = placement.rank()
        local_device = placement.device()

        _raise_if_mismatch(
            tensor_properties.layout,
            local_shard_tensor.layout,
            "layout",
            rank,
            True,
        )
        if not local_shard_tensor.is_contiguous():
            raise ValueError(
                "Only torch.contiguous_format memory_format is currently supported"
            )

        _raise_if_mismatch(
            shard_meta.shard_sizes,
            list(local_shard_tensor.size()),
            "size",
            rank,
        )
        _raise_if_mismatch(
            tensor_properties.pin_memory,
            local_shard_tensor.is_pinned(),
            "pin_memory",
            rank,
            True,
        )
        _raise_if_mismatch(local_device, local_shard_tensor.device, "device", rank)
        _raise_if_mismatch(
            tensor_properties.dtype,
            local_shard_tensor.dtype,
            "dtype",
            rank,
            True,
        )
        _raise_if_mismatch(
            tensor_properties.requires_grad,
            local_shard_tensor.requires_grad,
            "requires_grad",
            rank,
            True,
        )
    validate_non_overlapping_shards_metadata(shards_metadata)
    check_tensor(shards_metadata, list(sharded_tensor_metadata.size))
    sharded_tensor._local_shards = local_shards
    sharded_tensor._prepare_init(process_group=process_group,
                                 init_rrefs=init_rrefs)
    sharded_tensor._post_init()
    return sharded_tensor


def my_init_from_local_shards(
        cls, local_shards: List[Shard],
        *global_size, process_group=None, init_rrefs=False,
):
    process_group = cls._normalize_pg(process_group)
    current_rank = dist.get_rank(process_group)  # intentional to get global rank
    world_size = dist.get_world_size(process_group)

    local_sharded_tensor_metadata: Optional[ShardedTensorMetadata] = None
    global_tensor_size = utils._flatten_tensor_size(global_size)

    if len(local_shards) > 0:
        local_sharded_tensor_metadata = utils.build_metadata_from_local_shards(
            local_shards, global_tensor_size, current_rank, process_group
        )

    gathered_metadatas: List[Optional[ShardedTensorMetadata]] = []
    if world_size > 1:
        gathered_metadatas = [
            None for _ in range(world_size)
        ]

        dist.all_gather_object(gathered_metadatas, local_sharded_tensor_metadata,
                               group=process_group
                               )
    else:
        gathered_metadatas = [local_sharded_tensor_metadata]

    global_sharded_tensor_metadata = utils.build_global_metadata(gathered_metadatas)
    tensor_properties = global_sharded_tensor_metadata.tensor_properties

    spec = shard_spec._infer_sharding_spec_from_shards_metadata(
        global_sharded_tensor_metadata.shards_metadata
    )
    sharded_tensor = cls.__new__(
        cls,
        spec,
        global_sharded_tensor_metadata.size,
        dtype=tensor_properties.dtype,
        layout=tensor_properties.layout,
        pin_memory=tensor_properties.pin_memory,
        requires_grad=tensor_properties.requires_grad,
    )
    sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

    # attach local_shards to the ShardedTensor created
    sharded_tensor._local_shards = local_shards

    # run post initialization, i.e. map registration, rpc initialization
    sharded_tensor._post_init()
    return sharded_tensor


class BackgroundGenerator:
    """
    使用后台线程异步预取迭代器中的数据，以减轻主线程等待时间（例如避免卡在反序列化）。
    """

    def __init__(self, generator, max_prefetch=3):
        """
        :param generator: 一个 Python 迭代器，例如 map(...) 或 iter(dataloader)
        :param max_prefetch: 最大预取 batch 数，限制内存使用
        """
        self.generator = generator
        self.queue = queue.Queue(max_prefetch)
        self._thread = threading.Thread(target=self._worker)
        self._thread.daemon = True  # 主线程退出时，后台线程也自动结束
        self._thread.start()

    def _worker(self):
        try:
            for data in self.generator:
                self.queue.put(data)
        except Exception as e:
            self.queue.put(e)  # 异常传递到主线程
        finally:
            self.queue.put(None)  # 终止标志

    def __iter__(self):
        return self

    def __next__(self):
        next_item = self.queue.get()
        if isinstance(next_item, Exception):
            raise next_item  # 后台线程异常，转交主线程处理
        if next_item is None:
            raise StopIteration
        return next_item
