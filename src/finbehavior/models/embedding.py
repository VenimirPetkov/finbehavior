import torch
from torch import nn

from .config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int = (DEFAULT_EMBEDDING_DIMENSION),
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embedding(token_ids)
