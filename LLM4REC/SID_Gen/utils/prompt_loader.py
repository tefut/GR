# -*- coding: utf-8 -*-
"""
Prompt加载器 - 支持模板变量
将硬编码的System Prompt移至外部文件，支持Jinja2模板变量
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def load_prompt_file(prompt_path: str) -> str:
    """
    加载Prompt文件内容
    
    Args:
        prompt_path: Prompt文件路径
        
    Returns:
        str: prompt文本
    """
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt文件不存在: {prompt_path}")
    
    return path.read_text(encoding="utf-8")


def render_prompt_template(template: str, **kwargs: Any) -> str:
    """
    渲染Prompt模板
    
    Args:
        template: 模板文本
        **kwargs: 模板变量
        
    Returns:
        str: 渲染后的prompt文本
    """
    if kwargs:
        try:
            from jinja2 import Template
            return Template(template).render(**kwargs)
        except Exception as e:
            logger.warning("模板渲染失败，使用原始文本: %s", e)
            return template
    return template


def get_prompt_from_config(
    config: Dict[str, Any],
    prompt_type: str,
    reasoning_mode: str = "direct",
) -> str:
    """
    从配置获取Prompt（System Prompt）
    
    Args:
        config: 配置字典
        prompt_type: Prompt类型（如 "comment_summary", "app_desc", "clean_summary"）
        reasoning_mode: 推理模式
        
    Returns:
        str: prompt文本
    """
    prompts_config = config.get("prompts", {})
    prompt_base = prompts_config.get(prompt_type, {})
    
    if reasoning_mode == "think":
        prompt_path = prompt_base.get("think", f"prompts/{prompt_type}/think.txt")
    else:
        prompt_path = prompt_base.get("base", f"prompts/{prompt_type}/base.txt")
    
    # 获取项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    full_path = project_root / prompt_path
    
    try:
        return load_prompt_file(str(full_path))
    except FileNotFoundError:
        logger.warning("Prompt文件不存在: %s", full_path)
        return ""


def get_user_prompt_from_config(
    config: Dict[str, Any],
    prompt_type: str,
) -> str:
    """
    从配置获取User Prompt模板路径
    
    Args:
        config: 配置字典
        prompt_type: Prompt类型（如 "comment_summary", "app_desc", "clean_summary"）
        
    Returns:
        str: user prompt模板文件路径
    """
    prompts_config = config.get("prompts", {})
    prompt_base = prompts_config.get(prompt_type, {})
    user_path = prompt_base.get("user", f"prompts/{prompt_type}/user.txt")
    
    # 获取项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    full_path = project_root / user_path
    
    try:
        return load_prompt_file(str(full_path))
    except FileNotFoundError:
        logger.warning("User prompt模板文件不存在: %s", full_path)
        return ""


def get_preprocessor_config_from_config(
    config: Dict[str, Any],
    prompt_type: str,
) -> List[Dict[str, Any]]:
    """
    从配置获取预处理函数配置
    
    Args:
        config: 配置字典
        prompt_type: Prompt类型
        
    Returns:
        List[Dict[str, Any]]: 预处理函数配置列表
    """
    prompts_config = config.get("prompts", {})
    prompt_base = prompts_config.get(prompt_type, {})
    return prompt_base.get("preprocessors", [])


def get_placeholder_column_from_config(
    config: Dict[str, Any],
    prompt_type: str,
) -> str:
    """
    从配置获取占位符对应的列名
    
    Args:
        config: 配置字典
        prompt_type: Prompt类型
        
    Returns:
        str: 列名
    """
    prompts_config = config.get("prompts", {})
    prompt_base = prompts_config.get(prompt_type, {})
    return prompt_base.get("placeholder_column", "")


def get_output_column_from_config(
    config: Dict[str, Any],
    prompt_type: str,
) -> str:
    """
    从配置获取输出列名
    
    Args:
        config: 配置字典
        prompt_type: Prompt类型（如 "comment_summary", "app_desc", "clean_summary" ）
        
    Returns:
        str: 输出列名
    """
    prompts_config = config.get("prompts", {})
    prompt_base = prompts_config.get(prompt_type, {})
    return prompt_base.get("output_column", "")


def get_system_prompt_from_config(
    config: Dict[str, Any],
    prompt_type: str,
    reasoning_mode: str = "direct",
) -> str:
    """
    从配置获取 System Prompt 文本

    Args:
        config: 配置字典
        prompt_type: Prompt类型（如 "comment_summary", "app_desc", "clean_summary"）
        reasoning_mode: 推理模式（"direct" 或 "think"）

    Returns:
        str: system prompt文本
    """
    prompts_config = config.get("prompts", {})
    prompt_base = prompts_config.get(prompt_type, {})

    if reasoning_mode == "think":
        prompt_path = prompt_base.get("think", f"prompts/{prompt_type}/think.txt")
    else:
        prompt_path = prompt_base.get("base", f"prompts/{prompt_type}/base.txt")

    # 获取项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    full_path = project_root / prompt_path

    try:
        return load_prompt_file(str(full_path))
    except FileNotFoundError:
        logger.warning("System prompt文件不存在: %s", full_path)
        return ""


def list_available_prompts(prompt_dir: str) -> Dict[str, list]:
    """
    列出可用的Prompt
    
    Args:
        prompt_dir: Prompt目录路径
        
    Returns:
        dict: {prompt_name: [available_modes]}
    """
    result = {}
    prompt_path = Path(prompt_dir)
    
    if not prompt_path.exists():
        return result
    
    for subdir in prompt_path.iterdir():
        if subdir.is_dir():
            modes = []
            for f in subdir.glob("*.txt"):
                modes.append(f.stem)
            if modes:
                result[subdir.name] = modes
    
    return result
