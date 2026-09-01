import torch


def top_k_accuracy(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    k: int,
) -> float:
    if logits.ndim != 2:
        raise ValueError("Logits must have shape [batch_size, vocabulary_size]")

    if target_token_ids.ndim != 1:
        raise ValueError("Targets must have shape [batch_size]")

    if logits.shape[0] != target_token_ids.shape[0]:
        raise ValueError("Logits and targets must have the same batch size")

    vocabulary_size = logits.shape[1]

    if k <= 0 or k > vocabulary_size:
        raise ValueError("k must be between 1 and vocabulary size")

    top_k_token_ids = torch.topk(
        logits,
        k=k,
        dim=-1,
    ).indices

    target_matches = top_k_token_ids == target_token_ids.unsqueeze(-1)

    correct_predictions = target_matches.any(dim=-1)

    return correct_predictions.float().mean().item()
