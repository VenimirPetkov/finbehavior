import torch
from torch import nn

from .config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from .self_attention import SelfAttention


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        self_attention: SelfAttention,
        embedding_dimension: int = (DEFAULT_EMBEDDING_DIMENSION),
    ) -> None:
        super().__init__()

        self.self_attention = self_attention

        self.layer_norm = nn.LayerNorm(embedding_dimension)

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.self_attention(sequence)

        residual = sequence + attention_output

        return self.layer_norm(residual)
