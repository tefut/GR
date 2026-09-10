from typing import Dict, Optional

import torch


def _align_to_scores(value: torch.Tensor, scores: torch.Tensor) -> Optional[torch.Tensor]:
    if value.dim() == 1:
        value = value.unsqueeze(-1)
    if value.dim() == 3 and value.shape[-1] == 1:
        value = value.squeeze(-1)

    if value.shape == scores.shape:
        return value
    if value.dim() == 2 and value.shape[0] == scores.shape[0] and value.shape[1] == 1:
        return value.expand_as(scores)
    return None


def _get_positive_mask(
        model_input: Dict[str, torch.Tensor],
        key: str,
        scores: torch.Tensor
) -> Optional[torch.Tensor]:
    if key not in model_input:
        return None
    value = _align_to_scores(model_input[key], scores)
    if value is None:
        return None
    return value > 0


def apply_recent_paid_download_boost(
        scores: torch.Tensor,
        model_input: Dict[str, torch.Tensor],
        boost_conf: Dict,
        default_time_key: str
) -> torch.Tensor:
    """
    Boost the most recent paid game in the download/active segment.

    The CTR result is segmented: positions before ``start_rank`` keep the paid-game
    order, and positions from ``start_rank`` onward are sorted by download/active
    signals. For the tail segment, paid games with a download or active flag get a
    small score lift when their time feature is the closest/recent one.
    """
    if not boost_conf.get("enabled", False):
        return scores

    time_key = boost_conf.get("time_key", default_time_key)
    pay_flag_key = boost_conf.get("pay_flag_key", "is_pay_target_item_flag")
    down_flag_key = boost_conf.get("down_flag_key", "is_down_target_item_flag")
    use_flag_key = boost_conf.get("use_flag_key", "is_use_tagert_item_flag")
    if time_key not in model_input or pay_flag_key not in model_input:
        return scores

    candidate_time = _align_to_scores(model_input[time_key], scores)
    pay_mask = _get_positive_mask(model_input, pay_flag_key, scores)
    down_mask = _get_positive_mask(model_input, down_flag_key, scores)
    use_mask = _get_positive_mask(model_input, use_flag_key, scores)
    if candidate_time is None or pay_mask is None:
        return scores
    if down_mask is None and use_mask is None:
        return scores
    if down_mask is None:
        down_or_use_mask = use_mask
    elif use_mask is None:
        down_or_use_mask = down_mask
    else:
        down_or_use_mask = down_mask | use_mask

    start_rank = int(boost_conf.get("start_rank", 5))
    if start_rank >= scores.shape[1]:
        return scores

    pos = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
    tail_mask = pos >= start_rank
    valid_time_mask = candidate_time > 0
    candidate_mask = tail_mask & pay_mask & down_or_use_mask & valid_time_mask

    time_order = boost_conf.get("time_order", "desc")
    if boost_conf.get("time_is_distance", False):
        time_order = "asc"

    if time_order == "asc":
        masked_time = torch.where(
            candidate_mask,
            candidate_time,
            torch.full_like(candidate_time, torch.iinfo(candidate_time.dtype).max)
            if not candidate_time.dtype.is_floating_point
            else torch.full_like(candidate_time, float("inf"))
        )
        recent_time = masked_time.min(dim=1, keepdim=True).values
        boost_mask = candidate_mask & (masked_time == recent_time)
    else:
        masked_time = torch.where(
            candidate_mask,
            candidate_time,
            torch.zeros_like(candidate_time)
        )
        recent_time = masked_time.max(dim=1, keepdim=True).values
        boost_mask = candidate_mask & (masked_time == recent_time) & (recent_time > 0)

    boost_value = float(boost_conf.get("score_boost", 0.0))
    if boost_value == 0.0:
        return scores

    boosted_scores = scores + boost_mask.to(scores.dtype) * boost_value
    if not boost_conf.get("clamp_scores", True):
        return boosted_scores
    if boosted_scores.dtype == torch.float16:
        boosted_scores = torch.nan_to_num(boosted_scores, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.clamp(boosted_scores, min=0.0, max=1.0)