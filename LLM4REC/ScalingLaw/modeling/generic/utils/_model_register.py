from typing import Dict, Any
import logging

MODEL_CLS_DICT: Dict[str, Any] = {}


def register_model_cls(name=None):
    def decorator(cls):
        # 允许自定义名称，否则用类名
        model_name = name if name else cls.__name__
        if model_name in MODEL_CLS_DICT:
            logging.info("模型 %s 已经存在！", model_name)
        MODEL_CLS_DICT[model_name] = cls
        logging.info("注册模型：%s", model_name)
        return cls

    return decorator
