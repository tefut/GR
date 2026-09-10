# -*- coding: utf-8 -*-
"""
列名映射模块
支持多业务线的列名映射配置
"""

from typing import Any, Dict, Optional


class ColumnMapper:
    """列名映射器"""

    def __init__(self, mapping: Dict[str, str]):
        """
        初始化映射器

        Args:
            mapping: 列名映射字典 {标准名: 实际列名}
        """
        self.mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

    def get(self, standard_name: str, default: Optional[str] = None) -> Optional[str]:
        """获取标准名对应的实际列名"""
        return self.mapping.get(standard_name, default)

    def get_reverse(self, actual_name: str, default: Optional[str] = None) -> Optional[str]:
        """获取实际列名对应的标准名"""
        return self._reverse_mapping.get(actual_name, default)

    def items(self):
        """返回映射项"""
        return self.mapping.items()

    def __getitem__(self, key: str) -> Optional[str]:
        return self.mapping.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.mapping

    def __iter__(self):
        return iter(self.mapping)


def create_column_mapper(config: Dict[str, Any]) -> ColumnMapper:
    """
    从配置创建列映射器

    Args:
        config: 配置字典

    Returns:
        ColumnMapper: 列名映射器实例
    """
    col_mapping = config.get("column_mapping", {})
    columns = col_mapping.get("columns", {})

    return ColumnMapper(columns)
