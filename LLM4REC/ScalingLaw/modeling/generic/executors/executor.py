import abc
from typing import Tuple, Dict, Union, Iterable, Any

import torch


class Executor(abc.ABC):
    def __init__(self, config):
        self._config = config

    @abc.abstractmethod
    def evaluate_once(self, model_input: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluate once with the given model_input"""
        pass

    @abc.abstractmethod
    def execute(self, batch_id) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Execute with input datas initialized in `train(data_loader)` or `eval(data_loader)`"""
        pass

    @abc.abstractmethod
    def train(self, data_loader):
        """Should be called before training"""
        pass

    @abc.abstractmethod
    def eval(self, data_loader):
        """Should be called before eval"""
        pass

    @property
    @abc.abstractmethod
    def model(self):
        """Return the final executable model(DDP, DMP, ...)"""
        pass

    @abc.abstractmethod
    def gather_state_dict(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def reset(self):
        """Reset to initiating state only"""
        pass

    @abc.abstractmethod
    def close(self):
        """Release all resources"""
        pass

    def __del__(self):
        """Use `del` also called `close`"""
        self.close()

    @staticmethod
    def _del_dict_items(dct: Dict[Any, Any], keys: Iterable[Any]):
        for key in keys:
            if key in dct:
                del dct[key]

    @staticmethod
    def _del_objs(*objs):
        for obj in objs:
            if obj is not None:
                del obj

    def _get_optimizer(self, parameters: [Iterable[torch.Tensor] | Iterable[dict[str, Any]]]):
        optimizer_type = self._config["optimizer_type"].strip().lower()
        beta = tuple(self._config["beta"])
        learning_rate = self._config["learning_rate"]
        weight_decay = self._config["weight_decay"]

        if optimizer_type == "adamw":
            return torch.optim.AdamW(parameters, lr=learning_rate, betas=beta, weight_decay=weight_decay)
        elif optimizer_type == "adam":
            return torch.optim.Adam(parameters, lr=learning_rate, betas=beta, weight_decay=weight_decay)
        elif optimizer_type == "sgd":
            return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer_type %s" % optimizer_type)
