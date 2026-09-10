# -*- coding: utf-8 -*-
"""
Dead Code Resetter for RQ-VAE Codebook

Functions:
1. Track usage frequency of each codebook vector (based on EMA)
2. Periodically detect long-unused dead codes
3. Sample from current batch to replace dead codes
4. Support distributed training sync (via accelerator)

Tensor shapes:
    codebooks: [K, D] - codebook matrix, K=codebook size, D=embedding dim
    indices: [B] - selected indices for current batch
    z_e: [B, D] - encoder output (residual vector)

Distributed training design:
- usage_count: use all_reduce MEAN to get average usage frequency across ranks
- All ranks execute the same reset logic (dead_mask is identical after sync)
- No dependency on implicit DDP parameter sync for codebook changes
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeadCodeResetter(nn.Module):
    """
    Codebook reset module - detect and revive dead codes

    Codebook Collapse phenomenon:
    - In residual quantization, some codes are rarely selected, gradient≈0
    - EMA updates can only fine-tune active codes, cannot revive dead codes
    - Results in reduced vocabulary, increased collision rate

    Solution:
    - Periodically detect dead codes (usage < threshold)
    - Randomly sample from current batch to replace
    - Reset corresponding EMA state
    """

    def __init__(
            self,
            codebook: nn.Parameter,
            codebook_size: int,
            embed_dim: int,
            ema_decay: float = 0.99,
            reset_threshold: float = 1.0,
            reset_freq: int = 100,
            enable_reset: bool = True,
    ):
        """
        Args:
            codebook: codebook parameter [K, D]
            codebook_size: codebook size K
            embed_dim: embedding dimension D
            ema_decay: EMA decay factor, recommended 0.95~0.99
            reset_threshold: threshold for dead code detection
            reset_freq: reset frequency, execute every N steps
            enable_reset: whether to enable dead code reset
        """
        super().__init__()
        self.codebook = codebook
        self.K = codebook_size
        self.D = embed_dim
        self.ema_decay = ema_decay
        self.reset_threshold = reset_threshold
        self.reset_freq = reset_freq
        self.enable_reset = enable_reset

        # Non-gradient state buffers (must use register_buffer)
        # ema_cluster_dist: [K, D] EMA-updated codebook vectors
        self.register_buffer("ema_cluster_dist", torch.zeros_like(codebook.data))
        # usage_count: [K] usage count for each code (EMA weighted)
        self.register_buffer("usage_count", torch.zeros(self.K, dtype=torch.float32))
        # step_counter: training step counter
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))
        # seed_buffer: reserved for distributed-safe random seeding (currently using global_step as seed)

        # Initialize EMA with codebook values
        self.ema_cluster_dist.copy_(codebook.data)

        # Cache current batch's z_e for replacement
        self._cached_z_e: Optional[torch.Tensor] = None

    def cache_z_e(self, z_e: torch.Tensor) -> None:
        """
        Cache current batch's z_e for reset_dead_codes

        Args:
            z_e: encoder output [B, D]
        """
        self._cached_z_e = z_e.detach()

    def update_ema_from_indices(
            self,
            indices: torch.Tensor,
            z_e: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update EMA usage stats based on indices

        Args:
            indices: selected codebook indices [B]
            z_e: encoder output [B, D]
        """
        if z_e is not None:
            self.cache_z_e(z_e)

        with torch.no_grad():
            # Construct one-hot: [B, K]
            one_hot = F.one_hot(indices, num_classes=self.K).float()  # [B, K]
            # Update usage_count: [K]
            batch_counts = one_hot.sum(dim=0)  # [K]
            self.usage_count.mul_(self.ema_decay).add_(batch_counts, alpha=1 - self.ema_decay)

    @staticmethod
    def _get_world_size(accelerator) -> int:
        """Get number of processes, returns 1 for non-distributed"""
        if accelerator is None:
            return 1
        return getattr(accelerator, 'num_processes', 1)

    def _all_reduce_mean(self, tensor: torch.Tensor, accelerator) -> torch.Tensor:
        """
        All-reduce tensor across all ranks and compute mean.
        All ranks get the same result.
        """
        world_size = self._get_world_size(accelerator)
        if world_size <= 1:
            return tensor

        # Use all_reduce SUM then divide by world_size (equivalent to mean)
        tensor_sum = tensor.clone()
        torch.distributed.all_reduce(
            tensor_sum,
            op=torch.distributed.ReduceOp.SUM,
            async_op=False
        )
        return tensor_sum / world_size

    @torch.no_grad()
    def reset_dead_codes(
            self,
            z_e: Optional[torch.Tensor] = None,
            accelerator=None,
            global_step: int = None,
    ) -> Tuple[int, int]:
        """
        Detect and reset dead codes.

        Distributed design:
        - usage_count: sync via all_reduce MEAN across all ranks
        - All ranks compute identical dead_mask
        - All ranks execute the same replacement (same random seed)
        - No dependency on implicit DDP parameter sync

        Args:
            z_e: encoder output [B, D], will use cached if None
            accelerator: accelerate.Accelerator instance for distributed sync
            global_step: global training step for reset frequency check

        Returns:
            (num_dead, num_reset): detected dead codes, successfully reset codes
        """
        if not self.enable_reset:
            return 0, 0, {}

        # Use passed global_step or local step_counter
        step = global_step if global_step is not None else int(self.step_counter.item())

        # Check reset frequency - all ranks must use same global step
        if step % self.reset_freq != 0:
            if global_step is None:
                self.step_counter += 1
            return 0, 0, {}

        # Use passed z_e or cached
        if z_e is not None:
            self.cache_z_e(z_e)

        if self._cached_z_e is None:
            return 0, 0, {}

        world_size = self._get_world_size(accelerator)
        is_distributed = world_size > 1

        if is_distributed:
            self.usage_count.copy_(self._all_reduce_mean(self.usage_count, accelerator))

        # 2. All ranks compute identical dead_mask (same usage_count now)
        dead_mask = self.usage_count < self.reset_threshold  # [K]
        num_dead = int(dead_mask.sum().item())

        if num_dead == 0:
            return 0, 0, {}

        # 3. Gather all ranks' z_e for replacement sampling
        if is_distributed:
            all_z_e = accelerator.gather(self._cached_z_e)
        else:
            all_z_e = self._cached_z_e

        # 4. All ranks use the SAME random seed for sampling
        # Seed ensures all ranks select the same replacement vectors
        # global_step is guaranteed identical across ranks
        rng = torch.Generator(device=all_z_e.device)
        rng.manual_seed(int(step))

        B_total, D = all_z_e.shape
        if B_total == 0:
            return num_dead, 0, {}

        # batch_random strategy: all ranks sample the same dead indices using identical seed
        rand_idx = torch.randint(0, B_total, (num_dead,), device=all_z_e.device, generator=rng)
        samples = all_z_e[rand_idx].view(-1, D)  # [num_dead, D]

        # 5. All ranks execute identical replacement
        k_indices = torch.where(dead_mask)[0]  # dead code indices [num_dead]
        self.codebook.data[k_indices] = samples
        self.ema_cluster_dist[k_indices] = samples
        self.usage_count[k_indices] = 0.0

        # 6. Collect stats for logging
        total_usage = self.usage_count.sum().item()
        active_usage = self.usage_count[~dead_mask].sum().item()
        dead_usage_before = total_usage - active_usage
        stats = {
            "global_step": int(step),
            "num_dead": num_dead,
            "total_codes": self.K,
            "dead_ratio": num_dead / self.K,
            "reset_threshold": self.reset_threshold,
            "ema_decay": self.ema_decay,
            "dead_usage_before_reset": dead_usage_before,
            "active_usage": active_usage,
            "world_size": world_size,
        }

        return num_dead, num_dead, stats
