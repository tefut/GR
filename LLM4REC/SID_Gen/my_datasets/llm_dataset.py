# Copyright 2026-2026
"""
LLM任务数据集模块

提供统一的LLMDataset类，支持配置化的列名映射和有效行过滤。
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from torch.utils.data import Dataset


class LLMDataset(Dataset):
    """统一LLM任务数据集类
    
    支持从配置动态读取列名，实现跨业务复用。
    
    Args:
        df: 输入的DataFrame
        config: 任务配置字典，包含task、preprocessors等
        valid_indices: 可选的预过滤索引列表
    """
    
    def __init__(self, df: pd.DataFrame, config: Dict[str, Any], valid_indices: Optional[List[int]] = None):
        self.config = config
        task_config = config.get('task', {})
        
        # 从配置读取列名
        self.primary_key = task_config.get('primary_key', 'app_id')
        self.input_column = task_config.get('input_column', '')
        self.output_column = task_config.get('output_column', '')
        self.task_name = task_config.get('name', 'llm_task')
        
        # 获取有效条件
        self.valid_condition = task_config.get('valid_condition', 'not_empty')
        
        # 获取有效索引
        if valid_indices is not None:
            self.indices = valid_indices
        else:
            self.indices = self._get_valid_indices(df)
        
        # 提取数据
        self.ids = [df.iloc[i][self.primary_key] for i in self.indices]
        self.inputs = [df.iloc[i][self.input_column] for i in self.indices]
        
        # 记录总数
        self.total_count = len(df)
        self.valid_count = len(self.indices)
    
    def _get_valid_indices(self, df: pd.DataFrame) -> List[int]:
        """根据有效条件获取有效索引
        
        Args:
            df: 输入的DataFrame
            
        Returns:
            有效索引列表
        """
        if self.valid_condition == 'all':
            return list(range(len(df)))
        
        valid_indices = []
        for idx in range(len(df)):
            value = df.iloc[idx][self.input_column]
            
            if self.valid_condition == 'not_empty':
                # 非空判断
                if pd.notna(value) and str(value).strip():
                    valid_indices.append(idx)
            elif self.valid_condition == 'not_null':
                # 非None判断
                if pd.notna(value):
                    valid_indices.append(idx)
            else:
                # 默认非空判断
                if pd.notna(value) and str(value).strip():
                    valid_indices.append(idx)
        
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'id': self.ids[idx],
            'input': self.inputs[idx],
            'index': self.indices[idx]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息
        
        Returns:
            包含统计信息的字典
        """
        return {
            'task_name': self.task_name,
            'total_count': self.total_count,
            'valid_count': self.valid_count,
            'filter_rate': (self.total_count - self.valid_count) / max(self.total_count, 1),
            'primary_key': self.primary_key,
            'input_column': self.input_column,
            'output_column': self.output_column
        }
