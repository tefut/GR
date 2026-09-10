# 🚀 SID_Gen - 推荐义ID生成系统

**SID_Gen** 是一个基于 PyTorch 和 RQVAE 模型的推荐语义 ID（Semantic ID）生成系统。项目采用模块化设计，集成了 **LLM 内容增强
**、**Embedding 向量化**、**RQVAE 语义 ID 训练与评估** 大核心模块，提供从原始数据到语义 ID 的全链路标准化解决方案。

## ✨ 核心特性

- 🛠 **模块化架构**：数据处理、模型训练、推理评估高度解耦，支持灵活组合
- 🤖 **LLM 增强**：于 Accelerate 和 vLLM 的并行推理，支持多种 LLM 模型
- 📊 **多模态 Embedding**：支持纯文本及文本+图片多模态 Embedding 生成
- 🎯 **RQVAE 量化**：采用残差量化器（Residual Quantizer）生成多层语义 ID
- 📈 **多维评估**：提供码本容量、冲突率、熵、Gini 系数、NMI 等全面评估指标
- ⚡ **高效训**：支持多卡分布式训练、KMeans 初始化、死码自动重置等优化

---注：默认当通过MLOPS数据任务直接导入的DataOps表，即hive2nsp任务导入的txt文件，默认会以csv文件格式读取，需设置目标端分隔符为\u0001
才能正常读取，部分任务支持自定义分隔符，通过如sep超参数配置。

## 🧩 核心功能模块

### 模块一：LLM 生成模块 (`llm_generation/`)

本模块负责使用大语言模型对原始数据进行内容增强，支持应用描述生成、评论摘要、文本清洗等多种任务。

#### 📌 使用方式

**使用 Accelerate 多进程推理：**

```bash
python llm_generation/llm_generate.py \
    --config configs/llm/app_desc.yaml
```

**使用 vLLM 加速推理：**

```bash
python llm_generation/llm_generate_vllm.py \
    --config configs/llm/app_desc.yaml
```

#### ⚙️ 核心配置参数

| 参数名                         |   类型    |       默认值       | 详细说明                             |
|:----------------------------|:-------:|:---------------:|:---------------------------------|
| `input_file`                |  `str`  |     `None`      | **必填**。输入 CSV 文件路径               |
| `output_file`               |  `str`  |     `None`      | **必**。输出 CSV 文件路径                |
| `model_path`                |  `str`  |     `None`      | **必填**。LLM 模型路径或 HuggingFace 模型名 |
| `task.name`                 |  `str`  |      `""`       | 任务名称                             |
| `task.input_column`         |  `str`  | `"app_cn_name"` | 输入字段名                            |
| `task.output_column`        |  `str`  |  `"llm_desc"`   | 输出字段名                            |
| `task.primary_key`          |  `str`  |   `"app_id"`    | 主键列名                             |
| `task.valid_condition`      |  `str`  |  `"not_empty"`  | 数据有效性条件                          |
| `generation.max_new_tokens` |  `int`  |      `512`      | 生成的最大 Token 数量                   |
| `generation.temperature`    | `float` |      `0.7`      | 采样温度，越小越确定                       |
| `generation.top_p`          | `float` |      `0.9`      | Nucleus 采样阈值                     |
| `generation.max_length`     |  `int`  |     `8192`      | 输入最大 Token 长度                    |
| `generation.do_sample`      | `bool`  |     `True`      | 是否使用采样                           |
| `postprocessors`            | `list`  |      `[]`       | 后处理函数列表                          |
| `system_prompt_file`        |  `str`  |      `""`       | 系统 Prompt 模板文件路径                 |
| `user_prompt_file`          |  `str`  |      `""`       | 用户 Prompt 模板文件路径                 |

#### 📋 配置示例

**应用描述生 (`configs/llm/app_desc.yaml`)：**

```yaml
# 应用描述生成任务配置
input_file: "data/input.csv"
output_file: "data/output_with_desc.csv"
model_path: "/path/to/qwen3_4b_instruct"
aggregate: true

task:
  name: "app_desc"
  input_column: "app_cn_name"
  output_column: "llm_desc"
  primary_key: "app_id"
  valid_condition: "not_empty"

generation:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  do_sample: true
  max_length: 8192
  log_every_n_steps: 10

postprocessors:
  - name: "replace_empty"
    params:
      keywords:
        - "EMPTY_CONTENT"
        - "EMPTY"

system_prompt_file: "prompts/app_desc/base.txt"
user_prompt_file: "prompts/app_desc/user.txt"
```

**评论摘要生成 (`configs/llm/comment_summary.yaml`)：**

```yaml
# 评论摘要生成任务配置
input_file: "data/input.csv"
output_file: "data/output_with_summary.csv"
model_path: "/path/to/qwen3_4b_instruct"
aggregate: true

task:
  name: "comment_summary"
  input_column: "comments"
  output_column: "comment_summary"
  primary_key: "app_id"
  valid_condition: "not_empty"

generation:
  max_new_tokens: 1024
  temperature: 0.7
  top_p: 0.9
  do_sample: true
  max_length: 8192

postprocessors:
  - name: "replace_empty"
    params:
      keywords:
        - "EMPTY_CONTENT"
        - "EMPTY"

system_prompt_file: "prompts/comment_summary/base.txt"
user_prompt_file: "prompts/comment_summary/user.txt"
```

**摘要清洗 (`configs/llm/clean_summary.yaml`)：**

```yaml
# 摘要清洗任务配置
input_file: "data/output_with_summary.csv"
output_file: "data/output_cleaned.csv"
model_path: "/path/to/qwen3_4b_instruct"
aggregate: true

task:
  name: "clean_summary"
  input_column: "comment_summary"
  output_column: "cleaned_summary"
  primary_key: "app_id"
  valid_condition: "not_empty"

preprocessors:
  - name: "clean_text"
    params:
      keep_punctuation: true

generation:
  max_new_tokens: 1024
  temperature: 0.7
  top_p: 0.9
  do_sample: true

postprocessors:
  - name: "replace_empty"
    params:
      keywords:
        - "EMPTY_CONTENT"

system_prompt_file: "prompts/clean_summary/base.txt"
user_prompt_file: "prompts/clean_summary/user.txt"
```

---

### 模块二：Embedding 生成模块 (`embedding_generation/`)

本模块负责将文本数据转换为高维向量表示，支持纯文本及多模态（文本+图片）Embedding 成。

#### 📌 使用方式

```bash
python embedding_generation/emb_generate_txt.py \
    --config configs/embed_gen.yaml
```

或使用 shell 脚本：

```bash
bash run_scripts/run_embedding.sh --config=configs/embed_gen.yaml
```

#### ⚙️ 核心配置参数

| 参数名                        |   类型   |     默认值     | 详细说明                               |
|:---------------------------|:------:|:-----------:|:-----------------------------------|
| **数据输入**                   |        |             |                                    |
| `input_path`               | `str`  |   `None`    | **必填**。输入数据路径（支持 parquet/csv/txt）  |
| `output_path`              | `str`  |   `None`    | **必填**。输出 npz 文件路径                 |
| `input_columns`            | `list` |    `[]`     | 要拼接的本列名列表                          |
| `input_columns_desc`       | `list` |    `[]`     | 各列的描述信息（attach_desc 模式）            |
| `attach_desc`              | `bool` |   `False`   | 是否在拼接时附加列描述                        |
| `separator`                | `str`  |    `" "`    | 文本拼接分隔符                            |
| **型配置**                    |        |             |                                    |
| `model.plm_checkpoint`     | `str`  |   `None`    | **必填**。Embedding 模型路径              |
| `model.batch_size`         | `int`  |    `16`     | 推理批次大小                             |
| `model.max_sent_len`       | `int`  |   `1024`    | 最大文本长度（字符数）                        |
| `model.pooling`            | `str`  |  `"mean"`   | 池化策略：`mean`、`cls`、`last`           |
| `model.dtype`              | `str`  | `"float16"` | 数类型：`float16`、`bfloat16`、`float32` |
| **多模态配置**                  |        |             |                                    |
| `model.enable_multimodal`  | `bool` |   `False`   | 是否启用多模态（文本+图片）                     |
| `model.image_mapping_file` | `str`  |    `""`     | 图片路径映射文件                           |
| `model.image_base_dir`     | `str`  |    `""`     | 图片基础目录                             |

#### 📋 配置示例

**Embedding 生成配置 (`configs/embed_gen.yaml`)：**

```yaml
# Embedding生成任务专用配置
name: "embed_gen"
description: "Embedding生成任务配"

# 文本生成配置
text_generation:
  input_columns:
    - "app_cn_name"
    - "ctype"
    - "tags"
    - "app_second_type"
    - "app_third_type"
    - "app_desc"
    - "llm_desc"
  input_columns_desc:
    - "游戏名"
    - "应用内容类型"
    - "游戏标签"
    - "游戏二级分类"
    - "游戏三级分类"
    - "游戏描述"
    - "游戏描述"
  attach_desc: true
  separator: " "
  empty_placeholder: ""

# 图片映射配置
image_mapping:
  enabled: true
  mapping_file: "/path/to/id2imgpath.json"
  base_dir: "/path/to/game_icon"

# 模型配置
model:
  plm_checkpoint: "/path/to/qwen3_vl_embedding_2b_weights"
  plm_name: "qwen"
  batch_size: 16
  max_sent_len: 1024
  pooling: "mean"
  dtype: "float16"
  norm_embed: false
  enable_multimodal: true
  multimodal_mode: "text_image"

# 输出配置
output:
  input_path: "data/game_info_with_desc.parquet"
  output_path: "data/game_emb.npz"
  shard_dir: "data/emb_shards"
  save_shards: 1
  log_file: "emb_generate.log"
  save_csv: false

# 列名映射
column_mapping:
  primary_key: "app_id"

# 日志配置
log_level: "INFO"
```

#### 📤 输出格式

```python
npz = np.load("output.npz")
# npz["embedding"]: shape (N, embedding_dim)，Embedding 向量
# npz["app_id"]: shape (N,)，主键数组（可选）
```

---

### 模块三：RQVAE 训练与评估模块

#### 3.1 训练模块 (`train_sid/`)

基于残差量化变分自编码器（RQVAE）训练语义 ID 编码器。

##### 📌 使用方式

```bash
python train_sid/train_sid.py \
    --config configs/train_sid.yaml
```

或使用 shell 脚本：

```bash
bash run_scripts/run_train.sh --config=configs/train_sid.yaml
```

##### ⚙️ 核心配置参数

| 参数名                            |   类型    |                认值                 | 详细说明                                           |
|:-------------------------------|:-------:|:---------------------------------:|:-----------------------------------------------|
| **数据配置**                       |         |                                   |                                                |
| `data.data_path`               |  `str`  |              `None`               | **必填**。Embedding npz 文件路径                      |
| `data.data_type`               |  `str`  |              `"npz"`              | 数据格式：`npz`、`npy`、`h5`                          |
| `data.embed_key`               |  `str`  |           `"embedding"`           | npz 中 Embedding 的 key                          |
| `data.id_key`                  |  `str`  |            `"app_id"`             | npz 中 ID 的 key                                 |
| **训练配置**                       |         |                                   |                                                |
| `train.lr`                     | `float` |              `3e-4`               | 初始学习率                                          |
| `train.epochs`                 |  `int`  |              `2000`               | 训练轮数                                           |
| `train.batch_size`             |  `int`  |              `1024`               | 批大小                                            |
| `train.num_workers`            |  `int`  |                `4`                | DataLoader 并行数                                 |
| `train.eval_step`              |  `int`  |                `5`                | 评估间隔（epoch）                                    |
| `train.learner`                |  `str`  |             `"AdamW"`             | 优器：`AdamW`、`Adam`、`SGD`                        |
| `train.weight_decay`           | `float` |              `1e-6`               | 权重衰减系数                                         |
| `train.warmup_steps`           |  `int`  |                `0`                | 预热步数                                           |
| `train.lr_scheduler_type`      |  `str`  |            `"cosine"`             | 学习率调度器：`cosine`、`linear`、`constant`            |
| `train.min_lr`                 | `float` |              `1e-6`               | 最小学习率                                          |
| **模型配置**                       |         |                                   |                                                |
| `model.in_dim`                 |  `int`  |               `768`               | 输入 Embedding 维度                                |
| `model.num_emb_list`           | `list`  |         `[256, 256, 256]`         | 各层码本大小                                         |
| `model.e_dim`                  |  `int`  |               `32`                | 量化向量维度                                         |
| `model.layers`                 | `list`  | `[2048, 1024, 512, 256, 128, 64]` | MLP 网络层维度                                      |
| `model.dropout_prob`           | `float` |               `0.0`               | Dropout 概率                                     |
| `model.loss_type`              |  `str`  |              `"mse"`              | 损失类型：`mse`、`l1`                                |
| `model.recon_weight`           | `float` |               `1.0`               | 重构损失权重                                         |
| `model.quant_loss_weight`      | `float` |               `1.0`               | 量化损失权重                                         |
| `model.beta`                   | `float` |              `0.25`               | Commit Loss 权重                                 |
| `model.kmeans_init`            | `bool`  |              `True`               | 是否使用 KMeans 初始化码本                              |
| `model.sk_epsilons`            | `list`  |         `[0.0, 0.0, 0.0]`         | 各层 Sinkhorn epsilon                            |
| **死码重置**                       |         |                                   |                                                |
| `model.enable_dead_code_reset` | `bool`  |              `False`              | 是否启用死码重置                                       |
| `model.reset_threshold`        | `float` |               `1.0`               | 死码判定阈值（使用率 < 阈值视为死码）                           |
| `model.reset_freq`             |  `int`  |               `100`               | 死码重置率（步数）                                      |
| **输出配置**                       |         |                                   |                                                |
| `output.ckpt_dir`              |  `str`  |           `"./output"`            | Checkpoint 保存目录                                |
| `output.save_ckpt_mode`        |  `str`  |             `"last"`              | 保存模式：`last`、`best_loss`、`best_collision`、`all` |
| `output.export_item2sid`       | `bool`  |              `True`               | 是否导出 ID 到 SID 的映射 JSON                         |
| `output.export_sid_format`     |  `str`  |             `"token"`             | SID 格式：`token`（`<a_0>`）或 `index`（`[0, 0, 0]`）  |

#### 📋 配置示例

**RQVAE 训练配置 (`configs/train_sid.yaml`)：**

```yaml
# RQVAE训练任务专用配置
name: "train_sid"
description: "RQVAE训练任务配置"

column_mapping:
  primary_key: "app_id"

data_type: "npz"

# 训练配置
train_sid:
  # 数据和路径
  data_path: "data/game_emb.npz"
  ckpt_dir: "output/checkpoints"
  log_file: "train_sid.log"

  # 训练参数
  lr: 0.0003
  epochs: 2000
  batch_size: 1024
  num_workers: 4
  eval_step: 5
  learner: "AdamW"
  lr_scheduler_type: "cosine"
  warmup_epochs: 50
  weight_decay: 0.0

  # 模型参数
  dropout_prob: 0.0
  bn: false
  loss_type: "mse"
  kmeans_init: true
  kmeans_iters: 100
  sk_epsilons: [ 0.0, 0.0, 0.0 ]
  sk_iters: 50
  num_emb_list: [ 256, 256, 256 ]
  e_dim: 32
  quant_loss_weight: 1.0
  beta: 0.25
  layers: [ 2048, 1024, 512, 256, 128, 64 ]

  # 预训练
  pretrained_ckpt: ""

  # 评估输出
  save_limit: 5
  eval_dump_root: "output/sid"
  dump_sids: true
  dump_sids_format: "json"

  # 随机种子
  seed: 2024

# 设备配置
device: "cuda"

# 日志配置
log_level: "INFO"
```

##### 📤 训练输出

```
output/
├─ epoch_1000.pt              # 1000 epoch checkpoint
├── best_loss.pt               # 最佳 loss checkpoint
├─ best_collision.pt          # 最佳冲突率 checkpoint
└── item2sid.json              # ID 到 SID 的映射
```

##### item2sid.json 示例

```json
{
  "app_001": [
    "<a_187>",
    "<b_214>",
    "<c_187>"
  ],
  "app_002": [
    "<a_100>",
    "<b_055>",
    "<c_089>"
  ]
}
```

---

#### 3.2 评估模块 (`eval_sid/`)

对训练得到的语义 ID 进行多维度质量评估。

##### 📌 使用方式

```bash
python eval_sid/sid_eval_unified.py \
    --config configs/eval_sid.yaml
```

或使用 shell 脚本：

```bash
bash run_scripts/run_eval.sh --config=configs/eval_sid.yaml
```

##### ⚙️ 核心配置参数

| 参数名                   |   类型   |        默认值        | 详说明                   |
|:----------------------|:------:|:-----------------:|:----------------------|
| `paths.input_parquet` | `str`  |      `None`       | **必填**。输入数据 Parquet 路 |
| `paths.sid_file`      | `str`  |      `None`       | 包含 SID 的 Parquet 文件路径 |
| `primary_key`         | `str`  |    `"app_id"`     | 主键列名                  |
| `eval.sid_col`        | `str`  |      `"sid"`      | SID 列名                |
| `eval.name_col`       | `str`  |   `"app_name"`    | 名称列（用于日志）             |
| `eval.depth`          | `int`  |        `3`        | 量化深度（层数）              |
| `eval.vocab_sizes`    | `list` | `[256, 256, 256]` | 各层码本大小                |
| `eval.cat_cols`       | `list` |       `[]`        | 类目列（用于计算 NMI）         |

#### 📋 配置示例

**SID 评估配置 (`configs/eval_sid.yaml`)**

```yaml
# Eval SID 配置
name: "eval_sid"
description: "语义ID评估任务"

# 数据输入输出路径配置
paths:
  # 基础工作目录
  base_dir: "/path/to/work"

  # 输入 parquet 文件（包含 SID）
  input_parquet: "data/item_with_sid.parquet"

# 评价参数配置
eval:
  # SID 列名
  sid_col: "sid"
  name_col: "app_name"

  # 量化深度
  depth: 3

  # 各层词表大小
  vocab_sizes:
    - 256
    - 256
    - 256

  # 类目列名列表（用于计算 NMI）
  cat_cols:
    - "app_first_type"
    - "app_second_type"

# 主键列名
primary_key: "app_id"

# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s [%(levelname)s] %(message)s"
  log_file: "eval_sid.log"
```

##### 📊 评指标说明

| 指标类别        | 指标名                | 说明                                           |
|:------------|:-------------------|:---------------------------------------------|
| **码本容量**    | `Total Capacity`   | 总 SID 数量 = ∏ vocab_sizes                     |
| **冲突率**     | `Collision Rate`   | `(n - unique_sid) / n`，越低越好                  |
| **各层利用**    | `Layer X Usage`    | 已使用码字数 / 码本大小，越高越均匀                          |
| **信息熵**     | `Entropy (bits)`   | 各层码分布的信息熵，越高越均匀                              |
| **Gini 系数** | `Gini Coefficient` | 0-1 之间，越小越均匀                                 |
| **前缀突**     | `Prefix Collision` | L1 / L1-L2 前缀冲突率                             |
| **NMI**     | `NMI`              | Normalized Mutual Information，衡量 SID 与类目的一致性 |

---

## 🛠️ 工具功能脚本 (`scripts/`)

项目提供了一系列辅助脚本，用于数据处理和结果整合：

| 脚本名称                              | 功能描述                   | 使用示例                                                                            |
|:----------------------------------|:-----------------------|:--------------------------------------------------------------------------------|
| **`filter_content.py`**           | 过滤数据中的污染内容             | `python scripts/filter_content.py --input data.csv --output clean.csv`          |
| **`merge_app_desc.py`**           | 合并应用描述数据到主数据           | `python scripts/merge_app_desc.py --main data.csv --desc desc.csv`              |
| **`merge_app_sid_to_parquet.py`** | 将 SID 映射合并到 Parquet 文件 | `python scripts/merge_app_sid_to_parquet.py --data data.parquet --sid sid.json` |
| **`merge_csv_files.py`**          | 合并多个 CSV 文件            | `python scripts/merge_csv_files.py --inputs a.csv b.csv --output merged.csv`    |

#### 📋 工具脚本配置示例

**内容过滤配置 (`configs/filter_content.yaml`)：**

```yaml
# 内容过滤任务配置
name: "filter_content"
description: "内容过滤任务配置"
quoting: 'none'

# 文件路径
input_file: "data/input.csv"
output_file: "data/output_cleaned.csv"
sep: "\t"
input_column: "app_id"
filter_function: "register_value_filter"
filter_mode: "keep_true"
filter_params:
  filter_columns:
    - "image_embedding"
    - "pt_model"
    - "new_type"
  filter_values:
    - "NOT_EMPTY&!\\N"
    - "DINO_CLIP_WATCH"
    - "表盘"

# 日志配置
log_level: "INFO"
```

**相似度计算置 (`configs/compute_similarity.yaml`)：**

```yaml
# Embedding相似度计算任务配置

# 输入配置
input:
  embedding_file: "data/video_emb.npz"
  id_name_mapping_file: "data/video_info.csv"
  primary_key: "video_id"
  npz_emb_key: "embedding"
  name_column: "video_name"
  app_rank_file: ""

# 输出配置
output:
  output_csv: "data/similarity_result.csv"
  output_hive_csv: "data/similarity_result_hive.csv"
  top_n: 20
  visualize_rows: 10
  hive_top_n: 20

# 计算参数
chunk_size: 5000
top_k: 200
```

---

## 🚀 行脚本 (`run_scripts/`)

项目提供了封装好的运行脚本，支持快速启动各类任务：

| 脚本名称                    | 功能说明                 |
|:------------------------|:---------------------|
| `run_train.sh`          | RQVAE 训练任务           |
| `run_eval.sh`           | SID 评估任务             |
| `run_embedding.sh`      | Embedding 生成任务       |
| `llm_generate.sh`       | LLM 生成任务（Accelerate） |
| `llm_generate_vllm.sh`  | LLM 生成任务（vLLM 加速）    |
| `app_desc.sh`           | 应用述生成                |
| `comment_summary.sh`    | 评论摘要生成               |
| `clean_summary.sh`      | 摘要清洗                 |
| `merge_embeddings.sh`   | 合并 Embedding 文件      |
| `merge_csv_files.sh`    | 合并 CSV 文件            |
| `merge_app_desc.sh`     | 合并应用描述               |
| `merge_app_sid.sh`      | 合并 SID               |
| `compute_similarity.sh` | 计算 Embedding 相似度     |
| `filter_content.sh`     | 过滤内容                 |
| `add_csv_header.sh`     | 添加 CSV 表头            |
| `merge_desc.sh`         | 合并描述                 |
| `run_pipeline.sh`       | 完整流水线（包含所有步骤）        |

#### 使用方式

```bash
# 训练
bash run_scripts/run_train.sh --config=configs/train_sid.yaml

# 评估
bash run_scripts/run_eval.sh --config=configs/eval_sid.yaml

# Embedding 生成
bash run_scripts/run_embedding.sh --config=configs/embed_gen.yaml

# LLM 生成
bash run_scripts/llm_generate.sh --config=configs/llm/app_desc.yaml
```

---

## 📂 项目目录结构

```text
SID_Gen/
├── configs/                    # YAML 配置文件目录
│   ├── default.yaml            # 默认通用配置
│   ├── train_sid.yaml          # 训任务配置
│   ├── train_sid_picture.yaml  # 带图片的训练配置
│   ├── embed_gen.yaml          # Embedding 生成配置
│   ├── eval_sid.yaml           # 评估配置
│   ├── compute_similarity.yaml # 相度计算配置
│   ├── filter_content.yaml     # 内容过滤配置
│   ├── llm/                    # LLM 生成任务配置
│   │   ├── app_desc.yaml       # 应用描述生成配置
│   │   ├── comment_summary.yaml # 评论摘要生成配置
│   │   └── clean_summary.yaml  # 摘要清洗配置
│   └── column_mapping/         # 列名映射配置
│       └── game.yaml
├── llm_generation/             # LLM 内容生成模
│   ├── llm_generate.py         # Accelerate 多进程版本
│   ├── llm_generate_vllm.py    # vLLM 加版本
│   └── __init__.py
├── embedding_generation/       # Embedding 生成模块
│   └── emb_generate_txt.py     # 文本/多模态 Embedding 生成
├── train_sid/                  # RQVAE 训练模块
│   ├── train_sid.py            # 训练主脚本
│   ├── trainer.py              # Trainer 训练器实现
│   └── rqkmeans_plus.py        # KMeans 初始化增强
├── eval_sid/                   # 语义 ID 评估模块
│   ── sid_eval_unified.py     # 统一评估脚本
├── models/                     # 模型定义
│   ├── rqvae.py                # RQVAE 主模型
│   ├── rq.py                   # 残差量化器
│   ├── vq.py                   # 向量量化器
│   ├── layers.py               # 网络层组件
│   └── dead_code_resetter.py   # 死码重置器
├── my_datasets/                # 数据集实现
│   ├── emb_datasets.py         # Embedding 数据集
│   ├── llm_dataset.py          # LLM 数据集
│   └── __init__.py
├── prompts/                    # Prompt 模板目录
│   ├── app_desc/               # 应用描述 Prompt
│   ├── comment_summary/        # 评论摘要 Prompt
│   └── clean_summary/          # 摘要清洗 Prompt
├── scripts/                    # 工具功能脚本
│   ├── filter_content.py       # 内容过滤
│   ├── merge_app_desc.py       # 应用描述合并
│   ├── merge_app_sid_to_parquet.py # SID 合并
│   └── merge_csv_files.py      # CSV 文件合并
├── run_scripts/                # 任务运行脚本
│   ├── run_train.sh            # 练启动脚本
│   ├── run_eval.sh             # 评估启动脚本
│   └── ...                     # 其运行脚本
├── utils/                      # 工具模块
│   ├── config_loader.py        # 配置加载
│   ├── log_utils.py            # 日志工具
│   ├── preprocessor.py         # 预处理器
│   ├─ prompt_loader.py        # Prompt 加载器
│   └── column_mapping.py       # 列名映射
├── requirements.txt            # 依赖包列表
└── README.md                   # 项目说明文档
```

---

## 🔄 完整 Pipeline 流程

```
原始数据 (CSV/Parquet)
         ↓
┌──────────────────────────────────────┐
│ 1. LLM Generation                       │
│    - app_desc: 生成应用描述              │
│    - comment_summary: 评论摘要提取       │
│    - clean_summary: 文本清洗             │
└──────────────────────────────────────┘
         ↓ 增强后的数据
┌────────────────────────────────────────┐
│ 2. Embedding Generation                 │
│    - 文本拼接 + 模型编码                 │
│    - 可选：多模态（文本 + 图片）          │
└───────────────────────────────────────┘
         ↓ Embedding (npz)
┌──────────────────────────────────────┐
│ 3. Train SID (RQVAE)                    │
│    - 多层残差量化                        │
│    - 生成语义 ID                         │
└───────────────────────────────────────┘
         ↓ SID + Parquet
┌────────────────────────────────────────┐
│ 4. Eval SID                             │
│    - 码本容量评估                        │
│    - 冲突率评估                          │
│    - 熵/Gini 分析                        │
│    - NMI 计算                           │
└───────────────────────────────────────┘
         ↓ 最终语义 ID
```

---

## ⚙️ 配置优先级说明

```
命令行参数 > 业务配置文件 > 默认配置文件
```

例如：在 `configs/train_sid.yaml` 中设置的值会覆盖 `configs/default.yaml` 的默认值。

---

## 📦 环境赖

主要依赖包：

```
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
vllm>=0.2.0
numpy>=1.21.0
pandas>=1.5.0
pyarrow>=12.0.0
scikit-learn>=1.3.0
pyyaml>=6.0
```

详细依赖请参考 `requirements.txt`。

---

## 🎯 快速开始

### 1. 准备配置文件

根据任务需求修改 `configs/` 目录下的配置文件，或创建新的 YAML 配置文件。

### 2. 执行 LLM 增强（如需要）

```bash
python llm_generation/llm_generate.py \
    --config configs/llm/app_desc.yaml
```

### 3. 生成 Embedding

```bash
python embedding_generation/emb_generate_txt.py \
    --config configs/embed_gen.yaml
```

### 4. 训练语义 ID

```bash
python train_sid/train_sid.py \
    --config configs/train_sid.yaml
```

### 5. 评估结果

```bash
python eval_sid/sid_eval_unified.py \
    --config configs/eval_sid.yaml
```

---
