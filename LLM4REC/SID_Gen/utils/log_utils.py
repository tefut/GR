# -*- coding: utf-8 -*-
"""
日志工具模块
提供统一的日志初始化功能
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "root",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    获取或创建日志器
    
    Args:
        name: 日志器名称
        log_dir: 日志文件目录（可选）
        log_file: 日志文件名（可选）
        level: 日志级别
        format_str: 日志格式字符串
        
    Returns:
        logging.Logger: 配好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    # 设置日志格式
    if format_str is None:
        format_str = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    if log_dir and log_file:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    elif log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
