import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta=0.25,
                 kmeans_init=False, kmeans_iters=100, sk_iters=100,
                 enable_dead_code_reset=False,
                 reset_threshold=1.0,
                 reset_freq=100,
                 ema_decay=0.99,
                 ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        # Dead code reset 配置（统一传给每个 VQ 层）
        self.enable_dead_code_reset = enable_dead_code_reset
        self.reset_threshold = reset_threshold
        self.reset_freq = reset_freq
        self.ema_decay = ema_decay
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        beta=self.beta,
                                                        kmeans_init=self.kmeans_init,
                                                        kmeans_iters=self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters,
                                                        enable_dead_code_reset=enable_dead_code_reset,
                                                        reset_threshold=reset_threshold,
                                                        reset_freq=reset_freq,
                                                        ema_decay=ema_decay,
                                                        )
                                        for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices

    def reset_dead_codes(self, z_e: torch.Tensor, accelerator=None, global_step: int = None) -> dict:
        """
        触发所有 VQ 层死码重置

        Args:
            z_e: encoder 输出 [B, D]
            accelerator: accelerate.Accelerator 实例，用于分布式同步（可选）
            global_step: 全局训练步数（可选）

        Returns:
            dict: 各层的重置统计 {"layer_0": {"dead": 0, "reset": 0}, ...}
        """
        stats = {}
        for i, quantizer in enumerate(self.vq_layers):
            if quantizer.dead_code_resetter is not None:
                num_dead, num_reset, layer_stats = quantizer.dead_code_resetter.reset_dead_codes(
                    z_e, accelerator, global_step
                )
                stats[f"layer_{i}"] = {"dead": num_dead, "reset": num_reset, "detail": layer_stats}
            else:
                stats[f"layer_{i}"] = {"dead": 0, "reset": 0, "detail": {}}
        return stats
