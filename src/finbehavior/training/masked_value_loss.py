import torch
from torch.nn import functional as F


def masked_value_loss(
    logits: torch.Tensor,
    target_token_id: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        logits,
        target_token_id,
    )
