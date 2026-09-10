from model_registry import ModelRegistry
from model_initializer import ModelInitializer
import torch
import torch_npu
import os
from typing import Dict, List
import logging
import torch


class InferModeValidator:
    @staticmethod
    def valid(gr_module_cfg: Dict, inputs: List[Dict], infer_mode: str, error_threshold: float = 1e-6):
        """
        验证不同推理模式下模型输出的精度一致性，并生成误差统计报告。
 
        该静态方法通过固定随机种子确保实验可复现性，对比基准推理模式与目标推理模式的输出结果，
        从形状一致性和数值精度两两方面进行验证。若形状不一致直接报错，若数值误差超过阈值则统计为不一致样本，
        最终输出总样本数、最大/最小误差、不一致样本数及误差合格率等关键指标。
        方法执行完毕后会恢复原始随机数状态，避免影响外部环境的随机性。
 
        流程说明：
        1. 保存当前PyTorch随机数生成器状态，用于方法结束后恢复
        2. 固定随机种子为42，确保权重生成和推理过程的可复现性
        3. 生成基准模型权重并在基准模式下执行推理，获取基准结果列表
        4. 根据目标推理模式执行对应推理逻辑：
           - 若为"PD_SEP"模式，分"prefill"和"decode"两阶段推理，通过KV缓存传递中间结果
           - 其他模式直接执行单阶段推理，获取目标模式结果列表
        5. 结果验证与统计：
           - 首先检查基准结果与目标结果的形状是否一致，不一致则日志报错并返回
           - 累计总元素数量（通过numel()获取张量总元素数）
           - 计算绝对误差（abs_error）并更新最大/最小误差（转为Python数值）
           - 统计误差超过阈值的不一致元素数量（转为Python整数）
        6. 日志输出所有统计指标，包括总样本数、不一致数、误差范围及合格率
        7. 恢复原始随机数状态，确保不影响外部代码的随机性
 
        注意：
        - 形状不一致时会立即终止验证并返回
        - 所有统计指标均基于张量元素级别的比较（而非样本级）
        - 误差合格率计算公式：(1 - 不一致元素数 / 总元素数)
 
        Args:
            gr_module_cfg: 模型配置字典，包含模型结构、超参数等核心配置
            inputs: 输入数据列表，每个元素为包含输入特征的字典（支持批次推理）
            infer_mode: 目标推理模式（如"PD_SEP"、"prefill"、"decode"等），用于与基准模式对比
            error_threshold: 误差阈值，超过此值的样本判定为不一致，默认1e-6
        """
        original_state = torch.get_rng_state()
        torch.manual_seed(42)
 
        try:
            base_weight = InferModeValidator.__gen_base_weight(gr_module_cfg)
            base_result_list = InferModeValidator.__infer(gr_module_cfg, base_weight, inputs)
 
            if infer_mode == "PD_SEP":
                kv_list = InferModeValidator.__infer(gr_module_cfg, base_weight, inputs, "prefill")
                for idx, input_dict in enumerate(inputs):
                    input_dict["kv_cache"] = kv_list[idx]
                infer_mode_result_list = InferModeValidator.__infer(gr_module_cfg, base_weight, inputs, "decode")
            else:
                infer_mode_result_list = InferModeValidator.__infer(gr_module_cfg, base_weight, inputs, infer_mode)
 
            InferModeValidator.__show_statistics(base_result_list, infer_mode_result_list, error_threshold)
 
        except Exception as e:
            logging.error(f"Error during validation: {str(e)}")
            raise
        finally:
            torch.set_rng_state(original_state)
 
    @staticmethod
    def __show_statistics(base_result_list: List[torch.Tensor], 
                          infer_mode_result_list: List[torch.Tensor], error_threshold: float):
        non_consistence_count = 0
        max_error = -100
        min_error = 100
        total_elements = 0
        logging.info("----------------------------Consistence Statistics----------------------------")
        for base_result, infer_mode_result in zip(base_result_list, infer_mode_result_list):
            if base_result.shape != infer_mode_result.shape:
                logging.info("infer result shape is not consistence, base shape: %s, target shape: %s", 
                             base_result.shape, infer_mode_result.shape)
                return
            total_elements += base_result.numel()
            abs_error = torch.abs(base_result - infer_mode_result)
            max_error = max(torch.max(abs_error).item(), max_error)
            min_error = min(torch.min(abs_error).item(), min_error)
            non_consistence_count += torch.count_nonzero((abs_error > error_threshold).int()).item()
 
        satisfaction_rate = (1 - non_consistence_count / total_elements) * 100
        logging.info(f"Total elements: {total_elements}")
        logging.info(f"Non-consistent elements: {non_consistence_count}")
        logging.info(f"Minimum error: {min_error}")
        logging.info(f"Maximum error: {max_error}")
        logging.info(f"Error consistent rate: {satisfaction_rate:.4f}%")
        logging.info("----------------------------------------------------------------------------")
 
    @staticmethod
    def __gen_base_weight(gr_module_cfg: Dict):
        ModelRegistry.register_all_modules(modul_dir=os.path.join("./modeling/generic/sequential"))
        with InferModeMgr(gr_module_cfg):
            model = ModelInitializer.init(gr_module_cfg=gr_module_cfg)
            return model.state_dict()
 
    @staticmethod
    def __infer(gr_module_cfg: Dict, base_weight: Dict, inputs: List[Dict], infer_mode: str = None):
        with InferModeMgr(gr_module_cfg, infer_mode):
            model = ModelInitializer.init(gr_module_cfg=gr_module_cfg)
            model.load_state_dict(base_weight)
            model.to("npu")
            model.eval()
            return [model(input_dict) for input_dict in inputs]
 
 
class InferModeMgr:
    """
    推理模式上下文管理器，用于临时修改模型配置中的推理模式并自动恢复。
    """
    def __init__(self, gr_module_cfg: Dict, target_infer_mode: str = None):
        """
        初始化推理模式管理器。
 
        Args:
            gr_module_cfg (Dict): 模型配置字典，需包含"common_hp"键（通用超参配置）
            target_infer_mode (str, optional): 目标推理模式，默认为None（表示清除推理模式配置）
        """
        self.common_hp = gr_module_cfg["common_hp"]
        self.target_infer_mode = target_infer_mode
        # 记录原始推理模式（若配置中不存在则为None）
        self.origin_infer_mode = gr_module_cfg.get("common_hp", {}).get("infer_mode", None)
 
    def __enter__(self):
        """
        进入上下文时执行：将推理模式切换为目标模式。
        若目标模式为None，则从配置中移除推理模式键；否则设置为目标模式。
        """
        if self.target_infer_mode is None:
            self.common_hp.pop("infer_mode", None)
        else:
            self.common_hp["infer_mode"] = self.target_infer_mode
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文时执行：恢复原始推理模式。
        若原始模式为None，则从配置中移除推理模式键；否则恢复为原始模式。
 
        Args:
            exc_type: 异常类型（若有异常发生）
            exc_val: 异常值（若有异常发生）
            exc_tb: 异常追踪信息（若有异常发生）
        Returns:
            bool: 始终返回False，表示不抑制任何异常，让异常正常传播
        """
        if self.origin_infer_mode is None:
            self.common_hp.pop("infer_mode", None)
        else:
            self.common_hp["infer_mode"] = self.origin_infer_mode
        return False
