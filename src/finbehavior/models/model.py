import torch
from torch import nn

from finbehavior.tensorization.types import (
    TensorizedUser,
)

from .encoder import TransformerEncoder
from .user_sequence_embedding import (
    UserSequenceEmbedding,
)


class FinBehaviorModel(nn.Module):
    def __init__(
        self,
        user_sequence_embedding: UserSequenceEmbedding,
        encoder: TransformerEncoder,
    ) -> None:
        super().__init__()

        self.user_sequence_embedding = user_sequence_embedding

        self.encoder = encoder

    def forward(
        self,
        user: TensorizedUser,
    ) -> torch.Tensor:
        sequence = self.user_sequence_embedding(user)

        return self.encoder(sequence)
