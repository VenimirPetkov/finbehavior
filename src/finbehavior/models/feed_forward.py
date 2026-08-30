import torch
from torch import nn

from .config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from .config.feed_forward import (
    DEFAULT_FEED_FORWARD_EXPANSION_FACTOR,
)


class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_dimension: int = (DEFAULT_EMBEDDING_DIMENSION),
        expansion_factor: int = (DEFAULT_FEED_FORWARD_EXPANSION_FACTOR),
    ) -> None:
        super().__init__()

        hidden_dimension = embedding_dimension * expansion_factor

        self.input_projection = nn.Linear(
            embedding_dimension,
            hidden_dimension,
        )

        self.activation = nn.GELU()

        self.output_projection = nn.Linear(
            hidden_dimension,
            embedding_dimension,
        )

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(sequence)

        activated = self.activation(hidden)

        return self.output_projection(activated)
