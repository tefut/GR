import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm
from .dead_code_resetter import DeadCodeResetter


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim,
                 beta=0.25, kmeans_init=False, kmeans_iters=10,
                 sk_epsilon=0.003, sk_iters=100,
                 enable_dead_code_reset=False,
                 reset_threshold=1.0,
                 reset_freq=100,
                 ema_decay=0.99,
                 ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        # Dead code reset 配置
        self.enable_dead_code_reset = enable_dead_code_reset
        self.reset_threshold = reset_threshold
        self.reset_freq = reset_freq
        self.ema_decay = ema_decay

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        # 初始化 DeadCodeResetter（如果启用）
        self.dead_code_resetter: nn.Module = None
        if self.enable_dead_code_reset:
            self.dead_code_resetter = DeadCodeResetter(
                codebook=self.embedding.weight,
                codebook_size=self.n_e,
                embed_dim=self.e_dim,
                ema_decay=self.ema_decay,
                reset_threshold=self.reset_threshold,
                reset_freq=self.reset_freq,
                enable_reset=True,
            )

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5

        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, self.embedding.weight.t())
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)


            indices = torch.argmax(Q, dim=-1)



        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        # 更新 dead code reset 的 EMA（如果启用）
        # 在 forward 结束时更新这样与原始行为完全一致
        if self.dead_code_resetter is not None and self.training:
            self.dead_code_resetter.update_ema_from_indices(indices, latent)

        return x_q, loss, indices
