# -*- coding: utf-8 -*-
"""
配置 Schema 定义 - 使用 Pydantic 进行类型安全的配置验证
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class TextColumnConfig:
    """文本列配置"""
    name: str
    desc: str = ""
    max_len: int = 512


@dataclass
class ColumnMappingConfig:
    """列名映射配置"""
    primary_key: str = "app_id"
    columns: Dict[str, str] = field(default_factory=lambda: {
        "id": "app_id",
        "name": "app_cn_name",
        "description": "app_desc",
        "tags": "tags",
        "category_1": "app_first_type",
        "category_2": "app_second_type",
        "category_3": "app_third_type",
    })

    def get(self, logical_name: str) -> str:
        """获取逻辑名称对应的实际列名"""
        return self.columns.get(logical_name, logical_name)

    def get_primary_key(self) -> str:
        """获取主键列名"""
        return self.columns.get("id", self.primary_key)


@dataclass
class PreprocessorConfig:
    """预处理函数配置"""
    name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptTypeConfig:
    """单个Prompt类型的配置"""
    base: str = ""
    think: str = ""
    user: str = ""
    placeholder_column: str = ""
    preprocessors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PromptConfig:
    """Prompt配置"""
    comment_summary: PromptTypeConfig = field(default_factory=PromptTypeConfig)
    app_desc: PromptTypeConfig = field(default_factory=PromptTypeConfig)
    clean_summary: PromptTypeConfig = field(default_factory=PromptTypeConfig)


@dataclass
class EmbeddingConfig:
    """Embedding生成配置"""
    input_path: str = ""
    output_path: str = ""
    plm_checkpoint: str = "/opt/huawei/dataset/rqvae_models/Qwen1.5-1.8B"
    plm_name: str = "qwen"
    batch_size: int = 64
    max_sent_len: int = 512
    pooling: str = "mean"
    dtype: str = "float16"
    embedding_keys: List[str] = field(default_factory=lambda: [
        "app_cn_name", "tags", "app_second_type", "app_third_type", "app_desc"
    ])
    embedding_keys_desc: List[str] = field(default_factory=lambda: [
        "游戏名", "游戏标签", "游戏二级分类", "游戏三级分类", "游戏描述"
    ])
    append_name: bool = False
    word_drop_ratio: float = -1.0
    save_shards: int = 1
    shard_dir: str = ""
    enable_multimodal: bool = False
    multimodal_mode: str = "text_only"
    image_dir: str = ""
    norm_embed: bool = False


@dataclass
class TrainConfig:
    """训练配置"""
    data_path: str = ""
    ckpt_dir: str = ""
    lr: float = 5e-4
    epochs: int = 2000
    batch_size: int = 1024
    num_workers: int = 4
    eval_step: int = 5
    learner: str = "AdamW"
    lr_scheduler_type: str = "cosine"
    warmup_epochs: int = 50
    weight_decay: float = 0.0
    dropout_prob: float = 0.0
    num_emb_list: List[int] = field(default_factory=lambda: [256, 256, 256])
    e_dim: int = 32
    quant_loss_weight: float = 1.0
    beta: float = 0.25
    layers: List[int] = field(default_factory=lambda: [2048, 1024, 512, 256, 128, 64])
    pretrained_ckpt: str = ""
    save_limit: int = 5
    bn: bool = False
    loss_type: str = "mse"
    kmeans_init: bool = True
    kmeans_iters: int = 100
    sk_epsilons: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    sk_iters: int = 50
    eval_dump_root: str = "../../data"
    dump_sids: bool = True
    dump_sids_format: str = "json"
    # Dead code reset 配置
    enable_dead_code_reset: bool = False
    reset_threshold: float = 1.0
    reset_freq: int = 100
    ema_decay: float = 0.99


@dataclass
class EvalConfig:
    """评价配置"""
    data_path: str = ""
    sid_path: str = ""
    depth: int = 3
    vocab_sizes: List[int] = field(default_factory=lambda: [256, 256, 256])
    eval_dump_root: str = "../../data"


@dataclass
class BaseConfig:
    """基础配置 - 包含所有模块的通用配置"""
    name: str = "Sid_Gen"
    description: str = "Sid_Gen 语义ID生成系统"
    log_level: str = "INFO"
    log_file: str = ""
    device: str = "npu"
    debug_mode: bool = False
    debug_sample_size: int = 10
    use_8bit: bool = False

    # 列名映射
    column_mapping: ColumnMappingConfig = field(default_factory=ColumnMappingConfig)

    # Prompt配置
    prompts: PromptConfig = field(default_factory=PromptConfig)

    # 各模块配置
    embedding_generation: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    train_sid: TrainConfig = field(default_factory=TrainConfig)
    eval_sid: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """从字典创建配置"""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                if isinstance(value, dict):
                    existing = getattr(config, key)
                    if hasattr(existing, 'columns'):
                        # ColumnMappingConfig
                        for k, v in value.items():
                            if k == 'columns':
                                setattr(existing, k, v)
                            else:
                                setattr(existing, k, v)
                    else:
                        setattr(config, key, value)
                else:
                    setattr(config, key, value)
        return config
