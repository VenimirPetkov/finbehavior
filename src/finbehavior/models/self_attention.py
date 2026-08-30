import math

import torch
from torch import nn

from .config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)


class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dimension: int = (DEFAULT_EMBEDDING_DIMENSION),
    ) -> None:
        super().__init__()

        self.embedding_dimension = embedding_dimension

        self.query_projection = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False,
        )

        self.key_projection = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False,
        )

        self.value_projection = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False,
        )

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.query_projection(sequence)

        keys = self.key_projection(sequence)

        values = self.value_projection(sequence)

        attention_scores = torch.matmul(
            queries,
            keys.transpose(0, 1),
        )

        scaled_attention_scores = attention_scores / math.sqrt(self.embedding_dimension)

        attention_weights = torch.softmax(
            scaled_attention_scores,
            dim=-1,
        )

        return torch.matmul(
            attention_weights,
            values,
        )
