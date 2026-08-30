import torch
from torch import nn

from .config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)


class MaskedValuePredictionHead(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int = (DEFAULT_EMBEDDING_DIMENSION),
    ) -> None:
        super().__init__()

        if vocabulary_size <= 0:
            raise ValueError("Vocabulary size must be positive")

        self.output_projection = nn.Linear(
            embedding_dimension,
            vocabulary_size,
        )

    def forward(
        self,
        representation: torch.Tensor,
    ) -> torch.Tensor:
        return self.output_projection(representation)
