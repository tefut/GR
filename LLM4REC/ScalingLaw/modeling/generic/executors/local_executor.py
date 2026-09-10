import time
from typing import Optional, Tuple, Dict, Union

import torch
import torch.multiprocessing as mp
import torch_npu  # noqa
from torch.nn.parallel import DistributedDataParallel as DDP

from modeling.generic.executors.executor import Executor


class LocalExecutor(Executor):
    """Local executor"""

    def __init__(self, model, config, local_rank, device):
        super().__init__(config)
        self._model = model.to(torch.device(device))
        self._device = device
        self._batch_limit = config.get("batch_limit", 0)
        self._use_loss_weighted_grad = config.get("use_loss_weighted_grad", False)
        self._step_to_freeze_mlp = config.get("steps_to_freeze_mlp", False)
        self._mlp_frozen = False

        def _create_ddp(module):
            return DDP(
                module,
                device_ids=[local_rank],
                broadcast_buffers=False,
                find_unused_parameters=config.get("find_unused_parameters", False),
            )

        self._ddp_producer = _create_ddp
        self._ddp = self._ddp_producer(self._model)
        self._h2d_stream = torch_npu.npu.Stream(device)
        self._opt = None  # lazy init
        self._queue: Optional[mp.Queue] = None  # lazy init
        self._loader_proc: Optional[mp.Process] = None  # lazy init
        torch.set_num_threads(8)

    def _async_data_loader(self, data_loader, data_queue, is_train=True):
        for idx, row in enumerate(iter(data_loader)):
            if is_train and self._batch_limit and idx > self._batch_limit:
                break
            data_queue.put(row)

        data_queue.put("end")
        # Sleep to ensure the result of `queue.empty()` absolutely accurate
        time.sleep(1e-100)
        while not data_queue.empty():
            time.sleep(1)
        return

    def _execute_model(self, model_input: Dict[str, torch.Tensor]):
        if self._ddp.training:
            self._opt.zero_grad()
        output = self._ddp(model_input)
        if self._ddp.training:
            if self._use_loss_weighted_grad:
                output.backward(output)
            else:
                output.backward()
            self._opt.step()
        return output

    def evaluate_once(self, model_input: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        self._ddp.eval()
        return self._execute_model(model_input)

    def execute(self, batch_id) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        queue_data = self._queue.get()
        if queue_data == "end":
            self.reset()
            raise StopIteration
        model_input = {}
        with torch_npu.npu.stream(self._h2d_stream):
            for name, feature in queue_data.items():
                model_input[name] = feature.to(self._device, non_blocking=True)
        condition1 = self._ddp.training and self._step_to_freeze_mlp
        condition2 = not self._mlp_frozen and batch_id > self._step_to_freeze_mlp
        if (
            condition1 and condition2
        ):
            self._opt.zero_grad()
            for name, param in self._model.named_parameters():
                if "emb_mlp" in name or "feed_forward_" in name:
                    param.requires_grad = False
            self._ddp = self._ddp_producer(self._model)
            self._mlp_frozen = True

        output = self._execute_model(model_input)
        return output, model_input

    def train(self, data_loader):
        self._ddp.train()
        self._opt = self._get_optimizer(self._model.parameters())
        self._reset_queue()
        self._queue = mp.Queue(maxsize=3)
        self._loader_proc = mp.Process(target=self._async_data_loader, args=(data_loader, self._queue))
        self._loader_proc.start()

    def eval(self, data_loader):
        self._ddp.eval()
        self._reset_queue()
        self._queue = mp.Queue(maxsize=3)
        self._loader_proc = mp.Process(target=self._async_data_loader, args=(data_loader, self._queue, False))
        self._loader_proc.start()

    @property
    def model(self):
        return self._ddp

    def gather_state_dict(self, *args, **kwargs):
        return self._ddp.module.state_dict(*args, **kwargs)
        
    def _reset_queue(self):
        if self._loader_proc and self._loader_proc.is_alive():
            self._loader_proc.terminate()
            self._loader_proc.join(timeout=10)

            del self._loader_proc, self._queue
        self._loader_proc = None
        self._queue = None

    def reset(self):
        if self._opt is not None:
            self._opt.zero_grad(set_to_none=True)
            del self._opt
            self._opt = None
        self._reset_queue()

    def close(self):
        self.reset()
        self._del_objs(self._ddp, self._model, self._h2d_stream)
