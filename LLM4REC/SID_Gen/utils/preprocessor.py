# Copyright 2026 作者：灵犀
"""
预处理函数注册表
提供统一的预处理函数接口，支持通过配置调用
"""
import re
from functools import wraps
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

# 预处理函数注册表
PREPROCESSOR_REGISTRY: Dict[str, Callable] = {}

# 默认配置常量
DEFAULT_EMPTY_KEYWORDS = [
    "无法形成有效摘要",
    "缺乏可分析",
    "无法形成摘要",
]

# 预编译正则，大幅提升批量处理性能
_RE_CLEAN_STRICT = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]')
_RE_CLEAN_PUNCT = re.compile(
    r'[^\u4e00-\u9fa5'  # 汉字
    r'a-zA-Z0-9'  # 英文与数字
    r'\s'  # 空白符（空格/换行/制表）
    r'，。！？、；：“”‘’（）【】《》'  # 中文标点
    r'.,!?;:()\-\'\"/@#%&*+=]'  # 英文标点及常用符号
)
_RE_COLLAPSE_SPACE = re.compile(r'\s+')

DEFAULT_POLLUTION_PATTERNS = [
    r"---+[\s\S]*?生成[\s\S]*?摘要[\s\S]*?---+",
    r"---+[\s\S]*?请(根据|基于|按照)[\s\S]*?生成[\s\S]*?摘要",
    r"###\s*请(根据|基于|按照)[\s\S]*?生成[\s\S]*?摘要[\s\S]*?要求[\s\S]*?\)",
    r"###[\s\S]*?生成[\s\S]*?摘要",
    r"请(根据|基于|按照)[\s\S]*?生成[\s\S]*?摘要[\s\S]*?(输出要求|结构上遵循|字数不|核心优势|主要问题)",
    r"(输出要求|结构上遵循|严格按照|严格遵循|字数不|不要超过|不要提及|分析师|提取关键词|严禁编造|最终目标|输出规范|请仅输出|分析上述评论)",
    r"(AI 生成|是否符合|输出内容|【答案】|输出终稿|【注】|注意事项|概括陈述|最后提醒|用户画像分析|游客视角点评|最终结论)",
    r"^---+|---+$",
    r"Assistant[_\s]*Assistant",
    r"(Human|User|System|Assistant){2,}",
    r"assistant",
]


def register_preprocessor(name: str):
    """装饰器：注册预处理函数"""

    def decorator(func: Callable) -> Callable:
        PREPROCESSOR_REGISTRY[name] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_preprocessor(name: str) -> Callable:
    """获取预处理函数"""
    if name not in PREPROCESSOR_REGISTRY:
        raise ValueError(f"Unknown preprocessor: {name}. Available: {list(PREPROCESSOR_REGISTRY.keys())}")
    return PREPROCESSOR_REGISTRY[name]


def apply_preprocessors(text: str, preprocessors: List[Dict[str, Any]]) -> str:
    """按顺序应用预处理函数链
    
    Args:
        text: 输入文本
        preprocessors: 预处理配置列表，每个配置包含:
            - name: 预处理函数名
            - params: 参数字典（可选）
    
    Returns:
        处理后的文本
    """
    result = text
    for preprocessor_config in preprocessors:
        name = preprocessor_config.get("name", "")
        params = preprocessor_config.get("params", {})

        if not name:
            continue

        func = get_preprocessor(name)
        result = func(result, **params)

    return result


# ============ 基础文本处理函数 ============
@register_preprocessor("clean_text")
def clean_text(text: str, keep_punctuation: bool = True) -> str:
    """清理评论中的表情包、特殊符号、零宽字符，仅保留文本"""
    if not isinstance(text, str):
        return text  # 兼容 None/int 等异常类型

    # 1. 过滤非法字符
    pattern = _RE_CLEAN_PUNCT if keep_punctuation else _RE_CLEAN_STRICT
    cleaned = pattern.sub('', text)

    # 2. 合并多余空格（删除表情后常留下连续空白）
    cleaned = _RE_COLLAPSE_SPACE.sub(' ', cleaned).strip()
    return cleaned


# ============ 污染检测与清洗函数 ============

@register_preprocessor("replace_empty")
def replace_empty(text: str, keywords: List[str] = None, **kwargs) -> str:
    """检查文本是否为空摘要（包含空关键词）
    
    Args:
        text: 输入文本
        keywords: 空关键词列表，默认使用DEFAULT_EMPTY_KEYWORDS
    
    Returns:
        原文本（不做处理，仅用于检测）
    """
    if not keywords:
        keywords = DEFAULT_EMPTY_KEYWORDS

    for keyword in keywords:
        if keyword.lower() in text.lower():
            # 标记为空摘要
            return f""

    return text


@register_preprocessor("check_pollution")
def check_pollution(text: str, patterns: List[str] = None, **kwargs) -> str:
    """检查文本是否被系统Prompt污染
    
    Args:
        text: 输入文本
        patterns: 污染检测正则模式列表，默认使用DEFAULT_POLLUTION_PATTERNS
    
    Returns:
        原文本（不做处理，仅用于检测）
    """
    if not patterns:
        patterns = DEFAULT_POLLUTION_PATTERNS

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            # 标记为污染
            return f"[POLLUTED]{text}"

    return text


@register_preprocessor("clean_pollution")
def clean_pollution(text: str, patterns: List[str] = None, **kwargs) -> str:
    """清洗文本中的污染内容
    
    Args:
        text: 输入文本
        patterns: 污染检测正则模式列表，默认使用DEFAULT_POLLUTION_PATTERNS
    
    Returns:
        清洗后的文本
    """
    if not patterns:
        patterns = DEFAULT_POLLUTION_PATTERNS

    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    # 清理多余空白
    result = re.sub(r'\s+', ' ', result)
    return result.strip()


@register_preprocessor("truncate")
def truncate(text: str, max_length: int = 2000, **kwargs) -> str:
    """截断文本"""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


@register_preprocessor("filter_by_length")
def filter_by_length(text: str, min_length: int = 0, max_length: int = 100000, **kwargs) -> str:
    """根据长度过滤文本"""
    if not text:
        return ""
    length = len(text.strip())
    if length < min_length or length > max_length:
        return ""
    return text


# ============ 过滤函数（返回 bool） ============

def is_empty(text: Any) -> bool:
    """检查文本是否为空（None, NaN, 空字符串）"""
    if text is None:
        return True
    if isinstance(text, float) and np.isnan(text):
        return True
    if pd.isna(text):
        return True
    if isinstance(text, str) and not text.strip():
        return True
    return False


def is_not_empty(text: Any) -> bool:
    """检查文本是否非空"""
    return not is_empty(text)


def is_not_polluted(text: Any, patterns: List[str] = None) -> bool:
    """检查文本是否未被污染"""
    if is_empty(text):
        return False
    if not patterns:
        patterns = DEFAULT_POLLUTION_PATTERNS
    text = str(text)
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    return True


def is_length_valid(text: Any, min_length: int = 1, max_length: int = 100000) -> bool:
    """检查文本长是否在范围内"""
    if is_empty(text):
        return False
    text = str(text).strip()
    return min_length <= len(text) <= max_length


def parse_filter_condition(condition: str) -> tuple:
    """
    解析过滤条件字符串
    
    Args:
        condition: 条件字符串，支持以下格式：
            - "value1&value2" : 等于任意一个值（或者关系）
            - "!value1&value2" : 不等于任意一个值
            - "IS_EMPTY" : 值为空
            - "NOT_EMPTY" : 值不为空
            - "NOT_EMPTY&!value" : 值不为空且不等于value（组合条件）
            - "IS_EMPTY&value" : 值为空或等于value（组合条件）
    
    Returns:
        tuple: (operator, values)
            - operator: "eq", "ne", "is_empty", "not_empty" 或组合如 "not_empty&ne"
            - values: 值列表
    """
    condition = condition.strip()

    if "&" in condition:
        parts = condition.split("&")
        operators = []
        values = []
        for part in parts:
            part = part.strip()
            if part == "IS_EMPTY":
                operators.append("is_empty")
            elif part == "NOT_EMPTY":
                operators.append("not_empty")
            elif part.startswith("!"):
                operators.append("ne")
                values.append(part[1:])
            else:
                operators.append("eq")
                values.append(part)
        return ("&".join(operators), values)

    if condition == "IS_EMPTY":
        return ("is_empty", [])
    elif condition == "NOT_EMPTY":
        return ("not_empty", [])
    elif condition.startswith("!"):
        values_str = condition[1:]
        values = values_str.split("&")
        return ("ne", values)
    else:
        values = condition.split("&")
        return ("eq", values)


def check_condition(cell_value: str, operator: str, values: List[str]) -> bool:
    """
    检查单元格值是否满足条件
    
    Args:
        cell_value: 单元格的值
        operator: 操作符 ("eq", "ne", "is_empty", "not_empty") 或组合如 "not_empty&ne"
        values: 值列表
    
    Returns:
        bool: 是否满足条件
    """
    cell_value_str = str(cell_value) if cell_value is not None else ""
    is_empty_flag = cell_value_str.strip() == "" or cell_value is None or pd.isna(cell_value)

    if "&" in operator:
        operators = operator.split("&")
        idx = 0
        for op in operators:
            if op == "is_empty":
                if not is_empty_flag:
                    return False
            elif op == "not_empty":
                if is_empty_flag:
                    return False
            elif op == "eq":
                if idx >= len(values) or cell_value_str != values[idx]:
                    return False
                idx += 1
            elif op == "ne":
                if idx >= len(values) or cell_value_str == values[idx]:
                    return False
                idx += 1
        return True

    if operator == "is_empty":
        return is_empty
    elif operator == "not_empty":
        return not is_empty
    elif operator == "eq":
        return cell_value_str in values
    elif operator == "ne":
        return cell_value_str not in values
    else:
        return False


def filter_by_values(row, filter_columns, filter_values):
    """
    根据指定列的值进行过滤（支持多种条件）
    
    Args:
        row: DataFrame 的一行数据
        filter_columns: 要检查的列名列表
        filter_values: 对应的条件列表，与 filter_columns 一一对应
    
    Returns:
        bool: 所有列的条件都满足返回 True
    
    条件格式：
        - "value1&value2" : 等于任意一个值
        - "!value1&value2" : 不等于任意一个值
        - "IS_EMPTY" : 值为空
        - "NOT_EMPTY" : 值不为空
    """
    for col, condition in zip(filter_columns, filter_values):
        cell_value = row[col]
        operator, values = parse_filter_condition(condition)
        if not check_condition(cell_value, operator, values):
            return False
    return True


def register_value_filter(filter_columns: List[str], filter_values: List[str]):
    """
    注册一个基于列值的过滤器
    
    Args:
        filter_columns: 要检查的列名列表 ["status", "type", "name"]
        filter_values: 对应的条件列表，与 filter_columns 一一对应 ["active&pending", "!B", "NOT_EMPTY&!A"]
    Returns:
        function: 过滤函数
    """

    def filter_func(row):
        return filter_by_values(row, filter_columns, filter_values)

    return filter_func


__all__ = [
    "PREPROCESSOR_REGISTRY",
    "get_preprocessor",
    "apply_preprocessors",
    "is_empty",
    "is_not_empty",
    "is_not_polluted",
    "is_length_valid",
    "filter_by_values",
    "register_value_filter",
]
