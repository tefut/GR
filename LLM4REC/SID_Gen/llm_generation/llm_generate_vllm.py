"""
vLLM加LLM生成模块
支持vLLM批量推理、配置化预处理/生成/后处理流程
单进程运行，vLLM内部管理多卡（tensor_parallel）
"""

import argparse
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from SID_Gen.utils.log_utils import get_logger
from SID_Gen.utils.preprocessor import apply_preprocessors
from vllm import LLM, SamplingParams


# ============ 配置定义 ============

@dataclass
class LLMTaskConfig:
    """LLM任务配置"""
    name: str = ""
    input_column: str = ""
    output_column: str = ""
    primary_key: str = "app_id"
    valid_condition: str = "not_empty"

    preprocessors: List[Dict[str, Any]] = field(default_factory=list)

    generation: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 350,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "max_length": 8192,
        "log_every_n_steps": 0,
        "batch_size": 1,
        "reasoning_mode": "direct",
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 0,
        "quantization": "none",  # none, fp8, awq, gptq, bitsandbytes
    })

    postprocessors: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt_file: str = ""
    user_prompt_file: str = ""


# ============ 兼容修复 ============

def _patch_transformers_tokenizer():
    """修复 transformers >= 4.47 与 vLLM 的容性"""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    _original_getattr = getattr(PreTrainedTokenizerBase, '__getattr__', None)

    def _safe_getattr(self, name: str):
        if name == "all_special_tokens_extended":
            return getattr(self, 'all_special_tokens', [])
        if _original_getattr is not None:
            return _original_getattr(self, name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    PreTrainedTokenizerBase.__getattr__ = _safe_getattr


_patch_transformers_tokenizer()


# ============ 模型加载 ============

def detect_device_count() -> int:
    """自动检测可用设备数（NPU/CUDA）"""
    if torch.npu.is_available():
        count = torch.npu.device_count()
        print(f"Detected {count} NPU devices")
        return count
    elif torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"Detected {count} CUDA devices")
        return count
    else:
        print("No NPU/CUDA detected, using 1 device")
        return 1


def load_model(model_path: str, generation_params: Dict[str, Any], logger):
    """加载vLLM模型和tokenizer"""
    logger.info("Loading vLLM model from: %s", model_path)

    tp_size = generation_params.get("tensor_parallel_size", 0)
    if tp_size == 0:
        tp_size = detect_device_count()
    logger.info("Tensor parallel size: %d", tp_size)

    gpu_mem = generation_params.get("gpu_memory_utilization", 0.85)
    logger.info("GPU memory utilization: %.2f", gpu_mem)

    # 量化配置
    quantization = generation_params.get("quantization", "none")
    if quantization and quantization != "none":
        logger.info("Quantization: %s", quantization)
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_mem,
            max_model_len=generation_params.get("max_length", 8192),
            quantization=quantization,
        )
    else:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_mem,
            max_model_len=generation_params.get("max_length", 8192),
            enable_prefix_caching=False,
            dtype="float16",
        )

    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("vLLM model loaded successfully")
    return llm, tokenizer


# ============ Prompt构建 ============

def build_user_prompt(text: str, user_prompt_template: str = "") -> str:
    """构建用户Prompt"""
    if user_prompt_template:
        return user_prompt_template.replace("{PLACE_HOLDER}", text)
    return text


# ============ 后处理函数 ============

def apply_postprocessors(text: str, postprocessors: List[Dict[str, Any]]) -> str:
    """应用后处理函数链"""
    if not postprocessors:
        return text
    return apply_preprocessors(text, postprocessors)


def build_chat_template_kwargs(reasoning_mode: str, model_path: str) -> Dict[str, Any]:
    """构 chat_template_kwargs 参数"""
    kwargs = {}
    model_lower = model_path.lower()

    if "qwen3" in model_lower or "qwen-3" in model_lower:
        kwargs["enable_thinking"] = (reasoning_mode == "think")
    elif "deepseek" in model_lower:
        kwargs["enable_thinking"] = (reasoning_mode == "think")

    return kwargs


def build_sampling_params(generation_params: Dict[str, Any], reasoning_mode: str) -> SamplingParams:
    """构建 vLLM 采样参数，对齐 llm_generate.py 的参数"""
    max_tokens = generation_params.get("max_new_tokens", 350)

    if reasoning_mode == "think":
        max_tokens = max(max_tokens, 8192)
        temperature = generation_params.get("temperature", 0.6)
        top_p = generation_params.get("top_p", 0.95)
    else:
        temperature = generation_params.get("temperature", 0.7)
        top_p = generation_params.get("top_p", 0.9)

    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=20,
        repetition_penalty=1.1,
        min_p=0.0,
        stop_token_ids=None,  # eos_token_id 在 vLLM 由模型自动处理
    )


# ============ 生成函数 ============

def generate_results(
        llm,
        tokenizer,
        messages_batch: List[List[Dict]],
        indices: List[int],
        generation_params: Dict[str, Any],
        postprocessors: List[Dict[str, Any]],
        model_path: str,
):
    """用 llm.generate() 执行批量生成"""
    reasoning_mode = generation_params.get("reasoning_mode", "direct")
    sampling_params = build_sampling_params(generation_params, reasoning_mode)

    # 手动应用 chat template
    prompts = []
    enable_thinking = (reasoning_mode == "think")

    for messages in messages_batch:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        prompts.append(prompt)

    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    results = {}
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text or ""
        idx = indices[i]
        if generated_text and postprocessors:
            generated_text = apply_postprocessors(generated_text, postprocessors)
        results[idx] = generated_text

    return results


# ============ 文件处理 ============

def process_file(
        input_file: str,
        output_file: str,
        model_path: str,
        config: LLMTaskConfig,
        logger=None,
) -> bool:
    """处理单个文件（单进程，vLLM内部管理多卡）

    Returns:
        True: 正常处理完成
        False: 没有有效数据需要处理
    """
    logger.info("Processing file: %s", input_file)
    logger.info("Output file: %s", output_file)

    input_column = config.input_column

    df = pd.read_csv(input_file)
    logger.info("Loaded %d rows from CSV", len(df))

    if input_column not in df.columns:
        raise ValueError(f"Input column '{input_column}' not found in CSV. Available columns: {list(df.columns)}")

    valid_mask = df[input_column].notna() & (df[input_column].astype(str).str.strip() != "")
    valid_indices = df[valid_mask].index.tolist()
    logger.info("Valid rows after filtering: %d/%d", len(valid_indices), len(df))

    df_filtered = df[valid_mask].reset_index(drop=True)
    filtered_indices = list(range(len(df_filtered)))

    logger.info("Filtered data rows: %d", len(df_filtered))

    if len(df_filtered) == 0:
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        if config.output_column and config.output_column not in df.columns:
            df[config.output_column] = ""
        df.to_csv(output_file, index=False)
        logger.info("No valid data after filtering. Saved header to %s", output_file)
        return False

    generation_params = config.generation
    reasoning_mode = generation_params.get("reasoning_mode", "direct")

    llm, tokenizer = load_model(model_path, generation_params, logger)

    system_prompt = ""
    user_prompt_template = ""

    if config.system_prompt_file:
        system_prompt_path = Path(config.system_prompt_file)
        if system_prompt_path.exists():
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
            logger.info(f"Loaded system prompt from: {system_prompt_path}")

    if config.user_prompt_file:
        user_prompt_path = Path(config.user_prompt_file)
        if user_prompt_path.exists():
            with open(user_prompt_path, "r", encoding="utf-8") as f:
                user_prompt_template = f.read()
            logger.info(f"Loaded user prompt from: {user_prompt_path}")

    inference_batch_size = generation_params.get("batch_size", 1)
    log_every_n_steps = generation_params.get("log_every_n_steps", 0)

    all_results = {}
    total = len(df_filtered)
    step = 0

    for start_idx in range(0, total, inference_batch_size):
        step += 1
        end_idx = min(start_idx + inference_batch_size, total)

        messages_batch = []
        batch_indices = []

        for i in range(start_idx, end_idx):
            text = str(df_filtered.iloc[i][input_column])
            user_prompt = build_user_prompt(text, user_prompt_template)

            if config.preprocessors:
                user_prompt = apply_preprocessors(user_prompt, config.preprocessors)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            messages_batch.append(messages)
            batch_indices.append(i)

        batch_results = generate_results(
            llm=llm,
            tokenizer=tokenizer,
            messages_batch=messages_batch,
            indices=batch_indices,
            generation_params=generation_params,
            postprocessors=config.postprocessors,
            model_path=model_path,
        )

        all_results.update(batch_results)

        if log_every_n_steps > 0 and step % log_every_n_steps == 0:
            for si, st in batch_results.items():
                sl = len(st) if st else 0
                s_preview = (st[:100] + "...") if st and len(st) > 100 else (st or "(empty)")
                logger.info(f"  [Batch Result] idx={si}, len={sl}: {s_preview}")
            logger.info(f"Step {step}/{total // inference_batch_size + 1}: processed {end_idx}/{total}")

    logger.info("Saving %d results to: %s", len(all_results), output_file)
    save_results(df_filtered, all_results, output_file, config, input_file)
    logger.info("Processing completed")

    return True


def save_results(
        df_filtered: pd.DataFrame,
        all_results: Dict[int, str],
        output_file: str,
        config: LLMTaskConfig,
        input_file: str = None,
):
    """保存结果到CSV"""
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = df_filtered.copy()
    df_results[config.output_column] = ""

    for idx, text in all_results.items():
        if idx in df_results.index:
            df_results.at[idx, config.output_column] = text

    output_columns = [config.primary_key]
    if config.input_column and config.input_column in df_results.columns:
        output_columns.append(config.input_column)
    if config.output_column and config.output_column in df_results.columns:
        output_columns.append(config.output_column)
    df_output = df_results[output_columns]

    df_output.to_csv(output_file, index=False, encoding="utf-8")


# ============ 配置加载 ============

def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def parse_config(config_dict: Dict[str, Any]) -> LLMTaskConfig:
    """解析配置字为LLMTaskConfig"""
    task_config = config_dict.get("task", {})
    generation_config = config_dict.get("generation", {})

    generation = {
        "max_new_tokens": int(generation_config.get("max_new_tokens", 350)),
        "temperature": float(generation_config.get("temperature", 0.7)),
        "top_p": float(generation_config.get("top_p", 0.9)),
        "do_sample": bool(generation_config.get("do_sample", True)),
        "max_length": int(generation_config.get("max_length", 8192)),
        "log_every_n_steps": int(generation_config.get("log_every_n_steps", 0)),
        "batch_size": int(generation_config.get("batch_size", 1)),
        "reasoning_mode": generation_config.get("reasoning_mode", "direct"),
        "gpu_memory_utilization": float(generation_config.get("gpu_memory_utilization", 0.85)),
        "tensor_parallel_size": int(generation_config.get("tensor_parallel_size", 0)),
        "quantization": generation_config.get("quantization", "none"),
    }

    return LLMTaskConfig(
        name=task_config.get("name", ""),
        input_column=task_config.get("input_column", ""),
        output_column=task_config.get("output_column", ""),
        primary_key=task_config.get("primary_key", "app_id"),
        valid_condition=task_config.get("valid_condition", "not_empty"),
        preprocessors=config_dict.get("preprocessors", []),
        generation=generation,
        postprocessors=config_dict.get("postprocessors", []),
        system_prompt_file=config_dict.get("system_prompt_file", ""),
        user_prompt_file=config_dict.get("user_prompt_file", ""),
    )


# ============ 主函数 ============

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="vLLM LLM Generation Task")
    parser.add_argument("--config", type=str, required=True, help="Config file path (YAML format)")

    args = parser.parse_args()

    config_dict = load_config(args.config)

    input_file = config_dict.get('input_file')
    output_file = config_dict.get('output_file')
    model_path = config_dict.get('model_path')

    logger = get_logger("llm_generate_vllm", log_dir=Path(output_file).parent)

    if input_file is None or output_file is None or model_path is None:
        raise ValueError("配置文件中必须提供 input_file, output_file 和 model_path 参数")

    config = parse_config(config_dict)
    logger.info("Task config: %s", config)
    logger.info("tensor_parallel_size: %d (0=auto)", config.generation.get("tensor_parallel_size", 0))
    logger.info("batch_size: %d", config.generation.get("batch_size", 1))
    logger.info("primary_key: %s", config.primary_key)
    logger.info("input_column: %s", config.input_column)
    logger.info("output_column: %s", config.output_column)

    has_output = process_file(
        input_file=input_file,
        output_file=output_file,
        model_path=model_path,
        config=config,
        logger=logger
    )


if __name__ == "__main__":
    main()
