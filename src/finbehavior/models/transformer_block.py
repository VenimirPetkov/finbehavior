import torch
from torch import nn

from .config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from .feed_forward import FeedForward
from .self_attention_block import SelfAttentionBlock


class TransformerBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: SelfAttentionBlock,
        feed_forward: FeedForward,
        embedding_dimension: int = (DEFAULT_EMBEDDING_DIMENSION),
    ) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block

        self.feed_forward = feed_forward

        self.feed_forward_layer_norm = nn.LayerNorm(embedding_dimension)

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.self_attention_block(sequence)

        feed_forward_output = self.feed_forward(attention_output)

        residual = attention_output + feed_forward_output

        return self.feed_forward_layer_norm(residual)
