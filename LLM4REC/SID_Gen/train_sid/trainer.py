import json
import logging
import os
import stat
import string
from time import time

import numpy as np
import torch
from SID_Gen.utils.utils import ensure_dir, get_local_time, delete_file
from accelerate.utils import gather_object, broadcast_object_list
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, \
    get_cosine_schedule_with_warmup


def is_faiss_model(model):
    """
    判断模型是否为 FAISS 量化器对象

    rqkmeans 模式返回 dict，rqvae/rqkmeans_plus 返回 nn.Module

    Parameters:
        model: 模型实例

    Returns:
        bool: 是否为 FAISS 量化器
    """
    return isinstance(model, dict) and "rq" in model


class Trainer(object):
    """
    支持：
    1) 多卡/多进程训练（Accelerate）
    2) eval 计算 global collision_rate
    3) 可选 dump_sids：导出 item_id -> sid token list 的 JSON
    """

    def __init__(self, args, model, data_num, accelerator, wandb_run=None):
        self.args = args
        self.model = model
        self.accelerator = accelerator
        self.logger = logging.getLogger("SID_Gen")
        self.wandb_run = wandb_run

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs

        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []

        self.eval_step = min(args.eval_step, self.epochs)
        self.device = self.accelerator.device

        self.ckpt_dir = args.ckpt_dir
        self.eval_dump_root = args.eval_dump_root

        # 多卡下确保目录同步
        if self.accelerator.is_main_process:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            if self.eval_dump_root:
                os.makedirs(self.eval_dump_root, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_epoch = -1

        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self._prepared = False

        # 每多少步打印一次进度
        self.log_every_n_steps = getattr(args, 'log_every_n_steps', 100)

    # ----------------------------
    # helpers
    # ----------------------------
    def raw_model(self):
        return self.accelerator.unwrap_model(self.model)

    def _build_optimizer(self):
        params = self.model.parameters()
        learner = self.learner.lower()
        lr = self.lr
        wd = self.weight_decay

        if learner == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        if learner == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=wd)
        if learner == "adagrad":
            return optim.Adagrad(params, lr=lr, weight_decay=wd)
        if learner == "rmsprop":
            return optim.RMSprop(params, lr=lr, weight_decay=wd)
        if learner == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd)

        self.logger.warning("Unrecognized optimizer, default Adam")
        return optim.Adam(params, lr=lr)

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            return get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
            )
        elif self.lr_scheduler_type.lower() == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
            )
        else:
            return get_constant_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
            )

    @staticmethod
    def _check_nan(loss: torch.Tensor):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _format_collision(self, collision_rate: float) -> str:
        return f"{float(collision_rate):.6f}"

    def _log_dead_code_reset(self, epoch_idx: int, batch_idx: int, reset_stats: dict):
        """打印死码重统计信息，只在主进程调用"""
        lines = []
        total_dead = sum(v["dead"] for v in reset_stats.values())
        total_reset = sum(v["reset"] for v in reset_stats.values())
        details = [v.get("detail", {}) for v in reset_stats.values()]
        total_codes = sum(d.get("total_codes", 0) for d in details)
        total_ratio = total_dead / max(total_codes, 1)

        header = (f"[DeadReset][Epoch {epoch_idx}][Batch {batch_idx}] "
                  f"total_dead={total_dead}/{total_codes} (ratio={total_ratio:.3f}), total_reset={total_reset}")
        lines.append(header)

        for layer_name, layer_data in reset_stats.items():
            detail = layer_data.get("detail", {})
            if not detail:
                continue
            line = (
                f"  {layer_name}: dead={layer_data['dead']}/{detail.get('total_codes', 0)} "
                f"(ratio={detail.get('dead_ratio', 0):.3f}), "
                f"threshold={detail.get('reset_threshold', 0):.2f}, "
                f"ema_decay={detail.get('ema_decay', 0):.2f}, "
                f"world_size={detail.get('world_size', 1)}, "
                f"active_usage={detail.get('active_usage', 0):.1f}, "
                f"dead_usage={detail.get('dead_usage_before_reset', 0):.1f}, "
                f"step={detail.get('global_step', 0)}"
            )
            lines.append(line)

        self.logger.info("\n".join(lines))

    # ----------------------------
    # checkpoint save
    # ----------------------------
    def _save_checkpoint(self, epoch: int, collision_rate: float, tag: str = "epoch"):
        if not self.accelerator.is_main_process:
            return None

        cstr = self._format_collision(collision_rate)
        ckpt_name = f"{tag}.pt"
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)

        raw = self.raw_model()
        sd_cpu = {k: v.detach().cpu() for k, v in raw.state_dict().items()}

        state = {
            "args": vars(self.args),
            "epoch": int(epoch),
            "best_loss": float(self.best_loss),
            "best_collision_rate": float(self.best_collision_rate),
            "best_epoch": self.best_epoch,
            "state_dict": sd_cpu,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)
        self.logger.info(
            "Saving checkpoint: %s",
            ckpt_path
        )
        return ckpt_path

    # ----------------------------
    # training
    # ----------------------------
    def _train_epoch(self, train_data, epoch_idx: int):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0

        # 获取数据集长度
        num_batches = len(train_data)

        # 计算全局起始 step
        global_step_offset = epoch_idx * num_batches

        # 只在主进程打印
        if self.accelerator.is_main_process:
            self.logger.info(
                "[Train][Epoch %d] Start training with %d batches, logging every %d steps",
                epoch_idx, num_batches, self.log_every_n_steps
            )

        for batch_idx, batch in enumerate(train_data):
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                _ids, data = batch
            else:
                data = batch

            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if hasattr(data, "to"):
                data = data.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # 获取 encoder 输出用于 dead code reset
            raw_model = self.raw_model()
            if hasattr(raw_model, 'encoder'):
                z_e = raw_model.encoder(data)
            else:
                z_e = raw_model.mlp(data) if hasattr(raw_model, 'mlp') else None

            out, rq_loss, _indices = self.model(data)
            loss, loss_recon = raw_model.compute_loss(out, rq_loss, xs=data)

            self._check_nan(loss)

            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

            # 触发 dead code reset（在 optimizer.step 前）
            global_step = global_step_offset + batch_idx
            if getattr(self.args, 'enable_dead_code_reset', False) and global_step > 0:
                if hasattr(raw_model, 'rq') and hasattr(raw_model.rq, 'reset_dead_codes'):
                    reset_stats = raw_model.rq.reset_dead_codes(z_e.detach(), self.accelerator, global_step)
                    # 只在 reset_freq 周期执行时打印（total_dead 可为 0）
                    if self.accelerator.is_main_process and global_step % getattr(self.args, 'reset_freq', 100) == 0:
                        self._log_dead_code_reset(epoch_idx, batch_idx, reset_stats)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += float(loss.detach().item())
            total_recon_loss += float(loss_recon.detach().item())

            # 每 log_every_n_steps 步打印一次进度（只在主进程）
            if (batch_idx + 1) % self.log_every_n_steps == 0 or (batch_idx + 1) == num_batches:
                if self.accelerator.is_main_process:
                    progress = (batch_idx + 1) / num_batches * 100
                    avg_loss = total_loss / (batch_idx + 1)
                    avg_recon_loss = total_recon_loss / (batch_idx + 1)
                    self.logger.info(
                        "[Train][Epoch %d][%d/%d][%.1f%%] avg_loss: %.4f, avg_recon_loss: %.4f",
                        epoch_idx, batch_idx + 1, num_batches, progress, avg_loss, avg_recon_loss
                    )

        return total_loss, total_recon_loss

    # ----------------------------
    # SID dump utilities
    # ----------------------------
    @staticmethod
    def _encode_indices_to_token_lists(indices_np: np.ndarray):
        letters = list(string.ascii_lowercase)
        L = int(indices_np.shape[1])
        if L <= len(letters):
            prefixes = letters[:L]
        else:
            prefixes = [f"a{i}" for i in range(L)]
        out = []
        for row in indices_np:
            out.append([f"<{prefixes[j]}_{int(row[j])}>" for j in range(L)])
        return out

    def _dump_local_item2sid_shard(self, epoch_idx: int, pairs):
        shard_path = os.path.join(
            self.eval_dump_root,
            f"rank{self.accelerator.process_index:03d}.jsonl"
        )
        os.makedirs(os.path.dirname(shard_path), exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        modes = stat.S_IRUSR | stat.S_IWUSR
        with os.fdopen(os.open(shard_path, flags, modes), "w", encoding="utf-8") as f:
            for item_id, token_list in pairs:
                f.write(json.dumps({str(item_id): token_list}, ensure_ascii=False) + "\n")

    def _merge_item2sid_to_one_json(self, epoch_idx: int, out_name="item2sid.json"):
        if not self.accelerator.is_main_process:
            return None
        out_path = os.path.join(self.eval_dump_root, out_name)
        shard_files = []
        for fn in os.listdir(self.eval_dump_root):
            if fn.startswith("rank") and fn.endswith(".jsonl"):
                shard_files.append(os.path.join(self.eval_dump_root, fn))
        shard_files.sort()
        merged = {}
        for sp in shard_files:
            with open(sp, "r", encoding="utf-8") as r:
                for line in r:
                    line = line.strip()
                    if line:
                        merged.update(json.loads(line))
        merged = dict(sorted(merged.items(), key=lambda kv: kv[0]))
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        modes = stat.S_IRUSR | stat.S_IWUSR
        with os.fdopen(os.open(out_path, flags, modes), "w", encoding="utf-8") as w:
            json.dump(merged, w, ensure_ascii=False, indent=2)
        self.logger.info("Saved item->sid mapping: %s", out_path)
        return out_path

    # ----------------------------
    # validation
    # ----------------------------
    @torch.no_grad()
    def _valid_epoch(self, valid_data, epoch_idx: int):
        self.model.eval()
        should_dump = getattr(self.args, "dump_sids", False)

        num_batches = len(valid_data)

        # 只在主进程打印
        if self.accelerator.is_main_process:
            self.logger.info(
                "[Eval][Epoch %d] Start evaluation with %d batches, logging every %d steps",
                epoch_idx, num_batches, self.log_every_n_steps
            )

        local_indices_set = set()
        local_num_sample = 0
        dump_pairs = [] if should_dump else None

        for batch_idx, batch in enumerate(valid_data):
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                batch_ids, data = batch
            else:
                batch_ids, data = None, batch

            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if hasattr(data, "to"):
                data = data.to(self.device)

            bs = int(data.shape[0])
            local_num_sample += bs

            indices = self.raw_model().get_indices(data)
            indices = indices.view(-1, indices.shape[-1]).detach().cpu().numpy()

            for row in indices:
                local_indices_set.add("-".join([str(int(x)) for x in row]))

            if should_dump:
                if batch_ids is None:
                    raise ValueError("dump_sids=True but no item_id provided.")
                token_lists = self._encode_indices_to_token_lists(indices)
                dump_pairs.extend(list(zip(list(batch_ids), token_lists)))

            # 每 log_every_n_steps 步打印一次进度（只在主进程）
            if (batch_idx + 1) % self.log_every_n_steps == 0 or (batch_idx + 1) == num_batches:
                if self.accelerator.is_main_process:
                    progress = (batch_idx + 1) / num_batches * 100
                    self.logger.info(
                        "[Eval][Epoch %d][%d/%d][%.1f%%] processed_samples: %d",
                        epoch_idx, batch_idx + 1, num_batches, progress, local_num_sample
                    )

        local_unique_sid = len(local_indices_set)

        if self.accelerator.is_main_process:
            self.logger.info(
                "[Eval][local main] epoch=%d local_num_sample=%d local_unique_sid=%d",
                epoch_idx,
                local_num_sample,
                local_unique_sid,
            )

        gathered_sets = gather_object([local_indices_set])
        gathered_nums = gather_object([local_num_sample])
        gathered_uniqs = gather_object([local_unique_sid])

        if self.accelerator.is_main_process:
            global_set = set()
            for s in gathered_sets:
                global_set |= set(s)

            global_num = int(sum(gathered_nums))
            global_unique_sid = int(len(global_set))
            collision_rate = (global_num - global_unique_sid) / max(global_num, 1)

            self.logger.info(
                "[Eval][global] epoch=%d global_num=%d global_unique_sid=%d collision_num=%d collision_rate=%.6f",
                epoch_idx,
                global_num,
                global_unique_sid,
                global_num - global_unique_sid,
                collision_rate,
            )

            self.logger.info(
                "[Eval][per-rank] epoch=%d gathered_nums=%s gathered_unique_sids=%s",
                epoch_idx,
                gathered_nums,
                gathered_uniqs,
            )
        else:
            collision_rate = None

        collision_rate = broadcast_object_list([collision_rate], from_process=0)[0]
        collision_rate = float(collision_rate)

        if should_dump and collision_rate < self.best_collision_rate:
            self._dump_local_item2sid_shard(epoch_idx, dump_pairs)

            if self.accelerator.is_main_process:
                self.logger.info(
                    "[Eval][dump] epoch=%d local_dump_pairs=%d",
                    epoch_idx,
                    len(dump_pairs),
                )

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process and should_dump and collision_rate < self.best_collision_rate:
            self._merge_item2sid_to_one_json(epoch_idx)

        self.accelerator.wait_for_everyone()

        return collision_rate

    # ----------------------------
    # main loop
    # ----------------------------
    def fit(self, data_loader):
        if not self._prepared:
            self.model, self.optimizer, data_loader, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, data_loader, self.scheduler
            )
            self._prepared = True

        for epoch_idx in range(self.epochs):
            t0 = time()
            train_loss, train_recon_loss = self._train_epoch(data_loader, epoch_idx)
            t1 = time()

            if self.accelerator.is_main_process:
                self.logger.info(
                    self._generate_train_loss_output(epoch_idx, t0, t1, train_loss, train_recon_loss)
                )

                if self.wandb_run is not None:
                    self.wandb_run.log({
                        "train_loss": train_loss,
                        "train_recon_loss": train_recon_loss,
                        "epoch": epoch_idx,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    })

            if (epoch_idx + 1) % self.eval_step == 0:
                v0 = time()
                collision_rate = self._valid_epoch(data_loader, epoch_idx)
                v1 = time()

                if self.accelerator.is_main_process:
                    # 保存逻辑
                    ckpt_path = self._save_checkpoint(epoch_idx, collision_rate, tag="epoch")

                    if train_loss < self.best_loss:
                        self.best_loss = train_loss
                        self._save_checkpoint(epoch_idx, collision_rate, tag="best_loss")

                    if collision_rate < self.best_collision_rate:
                        self.best_collision_rate = collision_rate
                        self.best_epoch = epoch_idx
                        self._save_checkpoint(epoch_idx, collision_rate, tag="best_collision")

                    self.logger.info(
                        "[EvalStats] epoch=%d collision_rate=%.6f best_collision_rate=%.6f best_epoch=%d lr=%.8f",
                        epoch_idx,
                        collision_rate,
                        self.best_collision_rate,
                        self.best_epoch,
                        self.optimizer.param_groups[0]["lr"])

                    if self.wandb_run is not None:
                        self.wandb_run.log({
                            "collision_rate": collision_rate,
                            "best_collision_rate": self.best_collision_rate,
                            "best_loss": self.best_loss,
                            "best_epoch": self.best_epoch,
                            "eval_time": v1 - v0,
                            "epoch": epoch_idx,
                        })

                # 所有 Rank 在此同步
                self.accelerator.wait_for_everyone()

        return float(self.best_loss), float(self.best_collision_rate), self.best_epoch

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        out = "epoch %d training [time: %.2fs, " % (epoch_idx, e_time - s_time)
        out += "train loss: %.6f" % loss
        out += ", "
        out += "reconstruction loss: %.6f" % recon_loss
        return out + "]"
