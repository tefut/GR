import functools
import logging
import math
import time
from typing import Callable, Iterable, Optional, Any, Tuple

import torch
import torch.multiprocessing as mp
import torch_npu
from modeling.generic.sequential.GR_model import GR_model
from modeling.generic.sequential.features import seq_features_from_row_cpu, SequentialFeatures
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.npu.amp import autocast, GradScaler

# 类型别名：exec_cb 返回 (loss, seq_features) 的回调
ExecCbType = Callable[[Any, int], Tuple[torch.Tensor, 'SequentialFeatures']]

ASYNC_LOADER_FUNC = None


def _get_lr_lambda(warmup_steps, total_steps, schedule_type):
    """Return a lambda function for LambdaLR that implements warmup + decay schedule."""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if schedule_type == "step":
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if schedule_type == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif schedule_type == "linear":
            return max(0.0, 1.0 - progress)
        return 1.0

    return lr_lambda


def _create_lr_scheduler(opt, train_conf):
    """Create LR scheduler based on train_conf settings. Returns (scheduler, step_counter) or (None, None).

    lr_schedule_type:
      "constant" — warmup (if num_warmup_steps>0) then constant LR
      "cosine"   — warmup then cosine decay
      "linear"   — warmup then linear decay
      "step"     — warmup then constant (same as constant post-warmup)
    """
    schedule_type = train_conf.get("lr_schedule_type", "constant")
    warmup_steps = train_conf.get("num_warmup_steps", 0)
    total_steps = train_conf.get("total_training_steps", 0)

    # No warmup and constant schedule → no scheduler needed
    if warmup_steps <= 0 and schedule_type == "constant":
        return None, None

    # total_steps 未设置时，warmup 后 LR 保持恒定（等效 schedule_type="step"）
    if total_steps <= 0:
        logging.info("total_training_steps not set, using warmup-only schedule "
                     "(LR linearly warmup for %d steps, then constant).", warmup_steps)
        effective_type = "step"
    else:
        effective_type = schedule_type

    lr_lambda = _get_lr_lambda(warmup_steps, max(total_steps, warmup_steps), effective_type)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, [lr_lambda] * len(opt.param_groups))
    step_counter = [0]  # mutable list for closure state
    logging.info("LR scheduler: schedule=%s, warmup_steps=%d, total_steps=%d",
                 schedule_type, warmup_steps, total_steps)
    return scheduler, step_counter


def assemble_model_executable_callback(
        model: GR_model,
        optimizer_cb: Optional[Callable[[Iterable[torch.Tensor]], optim.Optimizer]],
        local_rank,
        device,
        train_conf,
        feature_conf,
        find_unused_parameters=False,
        amp_dtype="fp16"
):
    # Define async function
    torch.set_num_threads(8)

    def async_data_loader(data_loader, data_queue, include_candidate_items=False, done_event=None):
        gr_output_length = train_conf.get("gr_output_length", 0)
        batch_limit = train_conf.get("batch_limit", 0)
        for idx, row in enumerate(iter(data_loader)):
            if batch_limit and idx > batch_limit:
                break
            seq_feature_names = [k for k, _ in feature_conf['history_item_feature_columns'].items()]
            if not include_candidate_items:
                data = seq_features_from_row_cpu(
                    row, max_output_length=gr_output_length,
                    seq_item_feature_names=seq_feature_names,
                    user_feature_names=[k for k, _ in feature_conf['user_feature_columns'].items()],
                    include_loss_weights=True, itemid_column_name=feature_conf['candidate_items_key'],
                    infer_ratings_key=feature_conf["candidate_ratings_column"],
                    infer_timestamps_key=feature_conf["candidate_timestamps_column"]
                )
            else:
                data = seq_features_from_row_cpu(
                    row, max_output_length=gr_output_length + 1,
                    seq_item_feature_names=seq_feature_names,
                    user_feature_names=[k for k, _ in feature_conf['user_feature_columns'].items()],
                    include_loss_weights=True, itemid_column_name=feature_conf['candidate_items_key'],
                    include_candidate_items=include_candidate_items,
                    infer_ratings_key=feature_conf["candidate_ratings_column"],
                    infer_timestamps_key=feature_conf["candidate_timestamps_column"]
                )
            # share_memory_() enables zero-copy tensor sharing via mp.Queue
            for attr in ('uid', 'past_lengths', 'past_ids'):
                t = getattr(data, attr, None)
                if isinstance(t, torch.Tensor) and t.is_contiguous():
                    t.share_memory_()
            for t in data.past_payloads.values():
                if isinstance(t, torch.Tensor) and t.is_contiguous():
                    t.share_memory_()
            if data.past_embeddings is not None:
                if isinstance(data.past_embeddings, torch.Tensor) and data.past_embeddings.is_contiguous():
                    data.past_embeddings.share_memory_()
            data_queue.put(data)

        data_queue.put("end")
        # 保持进程存活直到主进程确认所有数据已消费完毕。
        # share_memory_()通过fd传递机制共享tensor，接收端data_queue.get()时需要
        # 连接回发送端进程的resource_sharer线程来rebuild_storage_fd。
        # 如果发送端进程先退出，接收端反序列化时FileNotFoundError。
        # 旧方案用while not data_queue.empty()不可靠（empty()不保证get()已完成）。
        if done_event is not None:
            done_event.wait()
        else:
            time.sleep(300)  # 兜底：保持进程存活5分钟

    # Set global func
    global ASYNC_LOADER_FUNC
    ASYNC_LOADER_FUNC = async_data_loader

    # 4. 初始化优化器
    opt = None
    if optimizer_cb is not None:
        opt = optimizer_cb(model.parameters())

    # 创建学习率调度器
    scheduler, scheduler_step_counter = _create_lr_scheduler(opt, train_conf) if opt is not None else (None, None)

    # 6. 训练模型
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=find_unused_parameters)

    h2d_stream = torch_npu.npu.Stream(device)
    use_loss_weighted_grad = train_conf.get("use_loss_weighted_grad", False)
    max_grad_norm = train_conf.get("max_grad_norm", 1.0)
    use_amp = train_conf.get("use_amp", False)
    _amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    # NPU FP16 scale管理
    if use_amp and amp_dtype != "bf16":
        _raw_scale = train_conf.get("loss_scale", 2 ** 16)
        init_scale = _raw_scale if _raw_scale > 0 else 2 ** 16
        dynamic_scale = train_conf.get("dynamic_loss_scale", True)
        scaler = GradScaler(
            init_scale=init_scale,
            growth_factor=2.0,
            backoff_factor=0.5,
            # 静态scale: growth_interval设极大值(100万)使训练过程中scale不增长,
            # 等效于关闭动态scale增长。Ascend GradScaler要求growth_factor>1.0,
            # 故不能设growth_factor=1.0, 只能通过拉长growth_interval来抑制增长。
            growth_interval=1000000 if not dynamic_scale else 100,
            enabled=True,
        )
    else:
        scaler = None

    def exec_cb(data_queue, batch_id):
        queue_data = data_queue.get()
        if queue_data == "end":
            _events = getattr(exec_cb, '_done_events', {})
            _evt = _events.get(data_queue)
            if _evt is not None:
                _evt.set()
            raise StopIteration

        payloads_tod = {}
        with torch_npu.npu.stream(h2d_stream):
            uid_device = queue_data.uid.to(device, non_blocking=True)
            historical_lengths_device = queue_data.past_lengths.to(device, non_blocking=True)
            for name, feature in queue_data.past_payloads.items():
                payloads_tod[name] = feature.to(device, non_blocking=True)

        seq_features = SequentialFeatures(
            uid=uid_device,
            past_lengths=historical_lengths_device,
            past_ids=payloads_tod.get(feature_conf.get('candidate_items_key')),
            past_payloads=payloads_tod,
        )

        if model.training:
            opt.zero_grad(set_to_none=True)

        model_input = seq_features.past_payloads
        model_input['past_lengths'] = seq_features.past_lengths
        model_input['past_ids'] = seq_features.past_ids
        model_input.pop("user_id", None)

        with autocast(enabled=use_amp, dtype=_amp_dtype):
            output = model(model_input=model_input)
        if model.training:
            if use_amp and scaler is not None:
                _scale_is_none = scaler._scale is None
                if _scale_is_none:
                    # NPU overflow信号：_scale被C++层置None表示检测到inf/nan梯度
                    # 执行backoff退避而非重置为1.0，防止scale单调递增
                    new_scale = init_scale * scaler.get_backoff_factor()
                    scaler._scale = torch.full(
                        (1,), new_scale, dtype=torch.float32, device=device
                    )
                    if scaler._growth_tracker is None:
                        scaler._growth_tracker = torch.full(
                            (1,), 0, dtype=torch.int64, device=device
                        )
                    opt.zero_grad(set_to_none=True)
                    # overflow步跳过optimizer step和scheduler step
                else:
                    scaler.scale(output).backward()
                    if max_grad_norm:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )
                    scaler.step(opt)
                    scaler.update()
                    # scaler.update()后若_scale为None，表示NPU在step中检测到overflow
                    # 执行backoff退避，且不推进scheduler
                    if scaler._scale is None:
                        new_scale = init_scale * scaler.get_backoff_factor()
                        scaler._scale = torch.full(
                            (1,), new_scale, dtype=torch.float32, device=device
                        )
                        if scaler._growth_tracker is None:
                            scaler._growth_tracker = torch.full(
                                (1,), 0, dtype=torch.int64, device=device
                            )
                        logging.warning(
                            "NPU overflow detected after scaler.update(), "
                            "backoff scale to %.1f", new_scale)
                    else:
                        # 正常步：推进scheduler
                        if scheduler is not None:
                            scheduler.step()
                            scheduler_step_counter[0] += 1
            else:
                # BF16 AMP (no scaler needed) or no-AMP path
                if use_loss_weighted_grad:
                    output.backward(output)
                else:
                    output.backward()
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                if scheduler is not None:
                    scheduler.step()
                    scheduler_step_counter[0] += 1
        return output, seq_features

    def _get_lr():
        """获取当前学习率，供外部日志使用"""
        if opt is not None and len(opt.param_groups) > 0:
            return opt.param_groups[0]['lr']
        return None

    exec_cb.get_lr = _get_lr

    return model, exec_cb


def assemble_ag_model_executable_callback(
        model: GR_model,
        optimizer_cb: Optional[Callable[[Iterable[torch.Tensor]], optim.Optimizer]],
        local_rank,
        device,
        train_conf,
        feature_conf,
        find_unused_parameters=False,
        amp_dtype="fp16"
):
    # Define async function
    torch.set_num_threads(8)

    def async_data_loader(data_loader, data_queue, done_event=None):
        for idx, row in enumerate(iter(data_loader)):
            if train_conf['batch_limit'] and idx > train_conf['batch_limit']:
                break
            # share_memory_() enables zero-copy tensor sharing via mp.Queue
            # instead of pickle serialization of storage bytes (major bottleneck)
            for v in row.values():
                if isinstance(v, torch.Tensor) and v.is_contiguous():
                    v.share_memory_()
            data_queue.put(row)

        data_queue.put("end")
        # 保持进程存活直到主进程确认所有数据已消费完毕（见AG版注释）
        if done_event is not None:
            done_event.wait()
        else:
            time.sleep(300)  # 兜底

    # Set global func
    global ASYNC_LOADER_FUNC
    ASYNC_LOADER_FUNC = async_data_loader

    # 4. 初始化优化器
    opt = None
    if optimizer_cb is not None:
        opt = optimizer_cb(model.parameters())

    # 创建学习率调度器
    scheduler, scheduler_step_counter = _create_lr_scheduler(opt, train_conf) if opt is not None else (None, None)

    # 6. 训练模型
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=find_unused_parameters)

    h2d_stream = torch_npu.npu.Stream(device)
    use_loss_weighted_grad = train_conf.get("use_loss_weighted_grad", False)
    max_grad_norm = train_conf.get("max_grad_norm", 1.0)
    use_amp = train_conf.get("use_amp", False)
    _amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    # NPU FP16 scale管理
    if use_amp and amp_dtype != "bf16":
        _raw_scale = train_conf.get("loss_scale", 2 ** 16)
        init_scale = _raw_scale if _raw_scale > 0 else 2 ** 16
        dynamic_scale = train_conf.get("dynamic_loss_scale", True)
        scaler = GradScaler(
            init_scale=init_scale,
            growth_factor=2.0,
            backoff_factor=0.5,
            # 静态scale: growth_interval设极大值(100万)使训练过程中scale不增长,
            # 等效于关闭动态scale增长。Ascend GradScaler要求growth_factor>1.0,
            # 故不能设growth_factor=1.0, 只能通过拉长growth_interval来抑制增长。
            growth_interval=1000000 if not dynamic_scale else 100,
            enabled=True,
        )
    else:
        scaler = None

    def exec_cb(data_queue, batch_id):
        queue_data = data_queue.get()
        if queue_data == "end":
            _events = getattr(exec_cb, '_done_events', {})
            _evt = _events.get(data_queue)
            if _evt is not None:
                _evt.set()
            raise StopIteration

        model_input = {}
        with torch_npu.npu.stream(h2d_stream):
            for name, feature in queue_data.items():
                model_input[name] = feature.to(device, non_blocking=True)

        if model.training:
            opt.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp, dtype=_amp_dtype):
            output = model(model_input=model_input, is_train=model.training)

        if model.training:
            if use_amp and scaler is not None:
                _scale_is_none = scaler._scale is None
                if _scale_is_none:
                    # NPU overflow信号：_scale被C++层置None表示检测到inf/nan梯度
                    # 执行backoff退避而非重置为1.0，防止scale单调递增
                    new_scale = init_scale * scaler.get_backoff_factor()
                    scaler._scale = torch.full(
                        (1,), new_scale, dtype=torch.float32, device=device
                    )
                    if scaler._growth_tracker is None:
                        scaler._growth_tracker = torch.full(
                            (1,), 0, dtype=torch.int64, device=device
                        )
                    opt.zero_grad(set_to_none=True)
                    # overflow步跳过optimizer step和scheduler step
                else:
                    scaler.scale(output).backward()
                    if max_grad_norm:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )
                    scaler.step(opt)
                    scaler.update()
                    # scaler.update()后若_scale为None，表示NPU在step中检测到overflow
                    # 执行backoff退避，且不推进scheduler
                    if scaler._scale is None:
                        new_scale = init_scale * scaler.get_backoff_factor()
                        scaler._scale = torch.full(
                            (1,), new_scale, dtype=torch.float32, device=device
                        )
                        if scaler._growth_tracker is None:
                            scaler._growth_tracker = torch.full(
                                (1,), 0, dtype=torch.int64, device=device
                            )
                        logging.warning(
                            "NPU overflow detected after scaler.update(), "
                            "backoff scale to %.1f", new_scale)
                    else:
                        # 正常步：推进scheduler
                        if scheduler is not None:
                            scheduler.step()
                            scheduler_step_counter[0] += 1
            else:
                if use_loss_weighted_grad:
                    output.backward(output)
                else:
                    output.backward()
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                if scheduler is not None:
                    scheduler.step()
                    scheduler_step_counter[0] += 1

        return output, model_input

    def _get_lr():
        """获取当前学习率，供外部日志使用"""
        if opt is not None and len(opt.param_groups) > 0:
            return opt.param_groups[0]['lr']
        return None

    exec_cb.get_lr = _get_lr

    return model, exec_cb


def feed_datas_to_exec_callback(exec_cb: ExecCbType, train_data_loader,
                                eval_data_loader) -> Tuple[Optional[ExecCbType], Optional[ExecCbType]]:
    """
    包装exec_cb为异步数据加载回调. 同时将exec_cb上的get_lr属性传递到functools.partial包装后的回调上,
    使外部(如train_step)可通过train_exec_cb.get_lr()获取当前学习率.
    """
    global ASYNC_LOADER_FUNC
    if ASYNC_LOADER_FUNC is None:
        raise RuntimeError("'feed_datas_to_exec_callback' must be called after 'assemble_model_executable_callback'")

    train_exec_cb = eval_exec_cb = None
    if train_data_loader is not None:
        train_data_queue = mp.Queue(maxsize=8)
        train_done_event = mp.Event()
        train_loader_proc = mp.Process(
            target=ASYNC_LOADER_FUNC,
            args=(train_data_loader, train_data_queue),
            kwargs={'done_event': train_done_event}
        )
        train_loader_proc.start()
        train_exec_cb = functools.partial(exec_cb, train_data_queue)
        # 传递get_lr到partial包装后的函数上，供日志获取学习率
        if hasattr(exec_cb, 'get_lr'):
            train_exec_cb.get_lr = exec_cb.get_lr
        # 将done_event设置到原始exec_cb函数上，StopIteration时自动通知loader进程退出
        # 注意：train和eval共用同一个exec_cb函数，但每个partial绑定不同的data_queue，
        # 需要用data_queue到done_event的映射来区分
        if not hasattr(exec_cb, '_done_events'):
            exec_cb._done_events = {}
        exec_cb._done_events[train_data_queue] = train_done_event

    if eval_data_loader is not None:
        eval_data_queue = mp.Queue(maxsize=8)
        eval_done_event = mp.Event()
        eval_loader_proc = mp.Process(
            target=ASYNC_LOADER_FUNC,
            args=(eval_data_loader, eval_data_queue),
            kwargs={'done_event': eval_done_event}
        )
        eval_loader_proc.start()
        eval_exec_cb = functools.partial(exec_cb, eval_data_queue)
        if hasattr(exec_cb, 'get_lr'):
            eval_exec_cb.get_lr = exec_cb.get_lr
        if not hasattr(exec_cb, '_done_events'):
            exec_cb._done_events = {}
        exec_cb._done_events[eval_data_queue] = eval_done_event

    return train_exec_cb, eval_exec_cb
