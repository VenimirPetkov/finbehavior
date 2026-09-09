import math

import torch
from torch import nn

from .config.embedding import DEFAULT_EMBEDDING_DIMENSION


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        super().__init__()

        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")

        self.embedding_dimension = embedding_dimension

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        if sequence.ndim != 2:
            raise ValueError("Sequence must have shape (length, embedding dimension)")

        if sequence.shape[-1] != self.embedding_dimension:
            raise ValueError(
                "Sequence embedding dimension does not match positional encoding"
            )

        positions = torch.arange(
            sequence.shape[0],
            device=sequence.device,
            dtype=torch.float32,
        ).unsqueeze(1)

        inverse_frequencies = torch.exp(
            torch.arange(
                0,
                self.embedding_dimension,
                2,
                device=sequence.device,
                dtype=torch.float32,
            )
            * (-math.log(10_000.0) / self.embedding_dimension)
        )

        angles = positions * inverse_frequencies.unsqueeze(0)

        encoding = torch.empty(
            (
                sequence.shape[0],
                self.embedding_dimension,
            ),
            device=sequence.device,
            dtype=torch.float32,
        )

        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])

        return sequence + encoding.to(dtype=sequence.dtype)
