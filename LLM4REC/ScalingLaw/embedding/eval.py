import torch
import torch.distributed as dist
from typing import Callable, Dict, List, Optional, Set
from embedding.indexing import CandidateIndex, TopKModule

from dataclasses import dataclass


@dataclass
class EvalState:
    all_item_ids: Set[int]
    candidate_index: CandidateIndex
    top_k_module: TopKModule


def eval_recall_metrics(
        eval_state: EvalState,
        model,
        model_input,
        target_ids: torch.Tensor,
        min_positive_rating: int = 1,
        epoch: Optional[str] = None,
        filter_invalid_ids: bool = True,
        user_max_batch_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
) -> Dict[str, List[float]]:
    B, _ = target_ids.shape
    device = target_ids.device
    MAX_K = 2500
    shared_input_embeddings = model.encode(
        past_lengths=model_input['past_lengths'],
        model_inputs=model_input,
    )
    k = min(MAX_K, eval_state.candidate_index.ids.size(1))
    user_max_batch_size = user_max_batch_size or shared_input_embeddings.size(0)
    eval_top_k_ids, eval_top_k_prs, _ = eval_state.candidate_index.get_top_k_outputs(
            query_embeddings=shared_input_embeddings,
            top_k_module=eval_state.top_k_module,
            k=k,
            invalid_ids=model_input['sequence_item_ids'] if filter_invalid_ids else None,
            return_embeddings=False,
        )
    _, eval_rank_indices = torch.max(
        torch.cat(
            [eval_top_k_ids, target_ids],
            dim=1,
        ) == target_ids,
        dim=1,
    )
    eval_ranks = torch.where(eval_rank_indices == k, MAX_K + 1, eval_rank_indices + 1)
    output = {
        "ndcg@1": torch.where(
            eval_ranks <= 1,
            1.0 / torch.log2(eval_ranks + 1),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "ndcg@10": torch.where(
            eval_ranks <= 10,
            1.0 / torch.log2(eval_ranks + 1),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "ndcg@50": torch.where(
            eval_ranks <= 50,
            1.0 / torch.log2(eval_ranks + 1),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "ndcg@100": torch.where(
            eval_ranks <= 100,
            1.0 / torch.log2(eval_ranks + 1),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "ndcg@200": torch.where(
            eval_ranks <= 200,
            1.0 / torch.log2(eval_ranks + 1),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "hr@1": (eval_ranks <= 1),
        "hr@10": (eval_ranks <= 10),
        "hr@50": (eval_ranks <= 50),
        "hr@100": (eval_ranks <= 100),
        "hr@200": (eval_ranks <= 200),
        "hr@500": (eval_ranks <= 500),
        "hr@1000": (eval_ranks <= 1000),
        "hr@2000": (eval_ranks <= 2000),
        "mrr": (1.0 / eval_ranks),
    }
    
    return output


def get_eval_state(
        all_item_ids: List[int],  # [X]
        negatives_sampler,
        top_k_module_fn: Callable[[torch.Tensor, torch.Tensor], TopKModule],
        device: torch.device,
        float_dtype: Optional[torch.dtype] = None,
) -> EvalState:
    # Exhaustively eval all items (incl. seen ids).
    eval_negatives_ids = torch.as_tensor(all_item_ids).to(device).unsqueeze(0)  # [1, X]
    eval_negative_embeddings = negatives_sampler.embedding_module.get_all_item_id_only_embeddings(eval_negatives_ids)
    #对所有item集合归一化
    eval_negative_embeddings = negatives_sampler.normalize_embeddings(eval_negative_embeddings)
    if float_dtype is not None:
        eval_negative_embeddings = eval_negative_embeddings.to(float_dtype)
    candidates = CandidateIndex(
        ids=eval_negatives_ids,
        embeddings=eval_negative_embeddings,
    )
    return EvalState(
        all_item_ids=set(all_item_ids),
        candidate_index=candidates,
        top_k_module=top_k_module_fn(eval_negative_embeddings, eval_negatives_ids)
    )
