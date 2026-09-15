import torch
from torch import nn

from finbehavior.tensorization.types import (
    TensorizedUser,
)

from .history_embedding import HistoryEmbedding
from .positional_encoding import SinusoidalPositionalEncoding
from .profile_embedding import ProfileEmbedding


class UserSequenceEmbedding(nn.Module):
    def __init__(
        self,
        profile_embedding: ProfileEmbedding,
        history_embedding: HistoryEmbedding,
    ) -> None:
        super().__init__()

        self.profile_embedding = profile_embedding
        self.history_embedding = history_embedding
        self.positional_encoding = SinusoidalPositionalEncoding(
            embedding_dimension=(profile_embedding.field_embedding.embedding_dimension),
        )

    def forward(
        self,
        user: TensorizedUser,
    ) -> torch.Tensor:
        profile_vector = self.profile_embedding(user.profile)

        history_vectors = self.history_embedding(user.events)

        sequence = torch.cat(
            (
                profile_vector.unsqueeze(0),
                history_vectors,
            ),
            dim=0,
        )

        return self.positional_encoding(sequence)
