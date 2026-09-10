# -*- coding: utf-8 -*-
"""
配置工具模块
提供从JSON文件加载配置的功能
"""

import json
import os


def load_config(path: str) -> dict:
    """
    读取标准 JSON 配置文件

    Args:
        path: JSON配置文件路径

    Returns:
        dict: 配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件不是有效的JSON对象
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件未找到: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("配置文件必须是一个 JSON 对象")
    return cfg
