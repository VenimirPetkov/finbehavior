import torch
from torch import nn

from .transformer_block import TransformerBlock


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        blocks: tuple[TransformerBlock, ...],
    ) -> None:
        super().__init__()

        if not blocks:
            raise ValueError("Encoder must contain at least one transformer block")

        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        encoded = sequence

        for block in self.blocks:
            encoded = block(encoded)

        return encoded
