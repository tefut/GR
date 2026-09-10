import torch
import torch.nn as nn


class Metrics(nn.Module):
    def __init__(self, top_k=200):
        super().__init__()
        self.top_k = max(top_k, 200)
        self.top_k_list = [1, 10, 50, 100, 200]

        self.reset()

    def reset(self):
        self.total_users = 0
        self.total_hits = {}
        self.total_ndcg = {}
        for top_k in self.top_k_list:
            hr_metric = f"hr@{str(top_k)}"
            self.total_hits[hr_metric] = 0.0

            ndcg_metric = f"ndcg@{str(top_k)}"
            self.total_ndcg[ndcg_metric] = 0.0

    def forward(self, preds, target_ids):
        '''
        计算当前批次的HR@K和NDCG@K
        :param preds: 模型预测分析，shape:[batch_size, top_k]
        :param targets: 用户实际交互的物品索引，shape:[batch_size, 1]
        '''
        self.total_users += target_ids.size(0)

        if target_ids.dim() == 1:
            target_ids = target_ids.unsqueeze(1)

        combined = torch.cat([preds, target_ids], dim=1)  # shape:[batch_size, k+1]

        _, eval_rank_indices = torch.max(combined == target_ids, dim=1)  # 第一个==target的索引位置

        eval_ranks = torch.where(
            eval_rank_indices == self.top_k,
            self.top_k + 1,
            eval_rank_indices + 1
        )

        hits = {}
        ndcg = {}
        for top_k in self.top_k_list:
            hr_metric = f"hr@{str(top_k)}"
            hr_score = (eval_ranks <= int(top_k)).float()
            self.total_hits[hr_metric] += hr_score.sum().item()
            hits[hr_metric] = hr_score.mean().item()

            ndcg_metric = f"ndcg@{str(top_k)}"
            ndcg_score = torch.where(
                eval_ranks <= int(top_k),
                1.0 / torch.log2(eval_ranks + 1),
                torch.zeros(1, dtype=torch.float32, device=preds.device),
            )
            self.total_ndcg[ndcg_metric] += ndcg_score.sum().item()
            ndcg[ndcg_metric] = ndcg_score.mean().item()

        return {"hits": hits, "ndcg": ndcg}

    def compute(self):
        """计算所有批次的累积指标"""
        if self.total_users == 0:
            self.reset()
            return {"hits": self.total_hits, "ndcg": self.total_ndcg}

        output_hits_metric = {}
        for metric in self.total_hits:
            output_hits_metric[metric] = self.total_hits[metric] / self.total_users

        output_ndcg_metric = {}
        for metric in self.total_ndcg:
            output_ndcg_metric[metric] = self.total_ndcg[metric] / self.total_users
        return {"hits": output_hits_metric, "ndcg": output_ndcg_metric}
