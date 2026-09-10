# -*- coding: utf-8 -*-
"""
配置加载器
支持默认配置 + 业务配置 + CLI 参数的合并优先级
"""

import os
from typing import Any, Dict, Optional

import yaml


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并配置字典

    Args:
        base: 基础配置
        override: 覆盖配置

    Returns:
        Dict: 合并后的配置
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def load_yaml_config(
        config_path: str,
        default_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    加载 YAML 配置文件，支持默认配置合并

    Args:
        config_path: 业务配置文件路径
        default_config_path: 默认配置文件路径（可选）

    Returns:
        Dict: 合并后的配置字典
    """
    default_cfg = {}

    # 加载默认配置
    if default_config_path and os.path.exists(default_config_path):
        with open(default_config_path, "r", encoding="utf-8") as f:
            default_cfg = yaml.safe_load(f) or {}

    # 加载业务配置
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            business_cfg = yaml.safe_load(f) or {}
        # 合并：业务配置覆盖默认配置
        return merge_configs(default_cfg, business_cfg)

    return default_cfg
