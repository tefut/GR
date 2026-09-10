# -*- coding: utf-8 -*-
"""
配置加载器 - 支持优先级覆盖
支持：默认配置 -> 业务覆盖配置 -> CLI参数
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

from .schema import BaseConfig, EmbeddingConfig, TrainConfig, EvalConfig

logger = logging.getLogger(__name__)


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """将嵌套字典展平为单层字典，键使用点号分隔"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json_config(config_path: str) -> Dict[str, Any]:
    """加载JSON配置文件"""
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config_file(config_path: str) -> Dict[str, Any]:
    """根据文件扩展名加载配置文件"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    suffix = path.suffix.lower()
    if suffix in [".yaml", ".yml"]:
        return load_yaml_config(config_path)
    elif suffix == ".json":
        return load_json_config(config_path)
    else:
        raise ValueError(f"不支持的配置文件格式: {suffix}")


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """合并多个配置字典，后者覆盖前者"""
    result = {}
    for cfg in configs:
        if cfg:
            result = _deep_merge(result, cfg)
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并两个字典"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    base_path: Optional[str] = None,
    override_path: Optional[str] = None,
    cli_args: Optional[argparse.Namespace] = None,
) -> Dict[str, Any]:
    """
    加载配置，支持三层优先级覆盖：
    默认配置 -> 业务覆盖配置 -> CLI参数
    
    Args:
        base_path: 默认配置路径（YAML格式）
        override_path: 业务覆盖配置路径（YAML格式）
        cli_args: CLI参数对象
        
    Returns:
        dict: 合并后的配置
    """
    configs = []
    
    # 1. 加载默认配置
    if base_path and Path(base_path).exists():
        try:
            default_cfg = load_config_file(base_path)
            configs.append(default_cfg)
            logger.info("加载默认配置: %s", base_path)
        except Exception as e:
            logger.warning("加载默认配置失败: %s", e)
    
    # 2. 加载业务覆盖配置
    if override_path and Path(override_path).exists():
        try:
            override_cfg = load_config_file(override_path)
            configs.append(override_cfg)
            logger.info("加载业务覆盖配置: %s", override_path)
        except Exception as e:
            logger.warning("加载业务覆盖配置失败: %s", e)
    
    # 合并配置
    merged = merge_configs(*configs) if configs else {}
    
    # 3. 应用CLI参数覆盖
    if cli_args:
        cli_cfg = _namespace_to_dict(cli_args)
        # 过滤掉None值
        cli_cfg = {k: v for k, v in cli_cfg.items() if v is not None}
        merged = _deep_merge(merged, cli_cfg)
        logger.info("应用CLI参数覆盖")
    
    return merged


def load_config_with_overrides(
    default_config: Optional[str] = None,
    business_config: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    加载配置并应用关键字参数覆盖
    
    Args:
        default_config: 默认配置路径
        business_config: 业务配置路径
        **overrides: 直接覆盖的配置项
        
    Returns:
        dict: 合并后的配置
    """
    cfg = load_config(base_path=default_config, override_path=business_config)
    
    # 应用关键字参数覆盖
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    
    return cfg


def _namespace_to_dict(ns: argparse.Namespace) -> Dict[str, Any]:
    """将Namespace对象转换为字典"""
    result = {}
    for key, value in vars(ns).items():
        if isinstance(value, argparse.Namespace):
            result[key] = _namespace_to_dict(value)
        else:
            result[key] = value
    return result


def args_to_config_dict(args: argparse.Namespace, prefix: str = "") -> Dict[str, Any]:
    """
    将CLI参数转换为配置字典（支持嵌套）
    
    Args:
        args: CLI参数对象
        prefix: 配置键前缀
        
    Returns:
        dict: 配置字典
    """
    result = {}
    for key, value in vars(args).items():
        if value is None:
            continue
        
        config_key = f"{prefix}{key}" if prefix else key
        
        # 处理嵌套的配置（如 embedding_generation.batch_size）
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_value is not None:
                    result[f"{config_key}.{sub_key}"] = sub_value
        else:
            result[config_key] = value
    
    return result


def create_base_config(config_dict: Dict[str, Any]) -> BaseConfig:
    """
    从配置字典创建BaseConfig对象
    
    Args:
        config_dict: 配置字典
        
    Returns:
        BaseConfig: 配置对象
    """
    return BaseConfig.from_dict(config_dict)


def get_module_config(config_dict: Dict[str, Any], module_name: str) -> Dict[str, Any]:
    """
    获取指定模块的配置
    
    Args:
        config_dict: 完整配置字典
        module_name: 模块名称（如 "embedding_generation", "train_sid"）
        
    Returns:
        dict: 模块配置
    """
    return config_dict.get(module_name, {})


def update_config_from_args(
    config: Dict[str, Any],
    args: argparse.Namespace,
    module_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    从CLI参数更新配置
    
    Args:
        config: 原始配置字典
        args: CLI参数对象
        module_name: 模块名称，如果指定则只更新该模块的配置
        
    Returns:
        dict: 更新后的配置
    """
    args_dict = _namespace_to_dict(args)
    
    if module_name:
        # 只更新指定模块的配置
        if module_name not in config:
            config[module_name] = {}
        config[module_name] = _deep_merge(config[module_name], args_dict)
    else:
        # 更新整个配置
        config = _deep_merge(config, args_dict)
    
    return config
