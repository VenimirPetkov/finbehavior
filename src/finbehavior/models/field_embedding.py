import torch
from torch import nn

from .config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from .embedding import TokenEmbedding


class FieldEmbedding(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int = (DEFAULT_EMBEDDING_DIMENSION),
    ) -> None:
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocabulary_size=vocabulary_size,
            embedding_dimension=embedding_dimension,
        )

    def forward(
        self,
        key_ids: torch.Tensor,
        value_ids: torch.Tensor,
    ) -> torch.Tensor:
        if key_ids.shape != value_ids.shape:
            raise ValueError("Key IDs and value IDs must have " "matching shapes")

        key_vectors = self.token_embedding(key_ids)

        value_vectors = self.token_embedding(value_ids)

        return key_vectors + value_vectors
