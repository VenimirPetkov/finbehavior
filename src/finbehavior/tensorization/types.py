from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorizedProfile:
    user_token_id: torch.Tensor
    key_ids: torch.Tensor
    value_ids: torch.Tensor
