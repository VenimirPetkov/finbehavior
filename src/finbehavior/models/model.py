import torch
from torch import nn

from finbehavior.tensorization.types import (
    TensorizedUser,
)

from .encoder import TransformerEncoder
from .sequence_batch import (
    build_sequence_batch,
)
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

    def encode_sequence(
        self,
        user: TensorizedUser,
    ) -> torch.Tensor:
        sequence = self.user_sequence_embedding(user)

        return self.encoder(sequence)

    def encode_users(
        self,
        users: tuple[TensorizedUser, ...],
    ) -> torch.Tensor:
        if not users:
            raise ValueError("User batch must contain at least one user")

        sequences = tuple(self.user_sequence_embedding(user) for user in users)

        sequence_batch = build_sequence_batch(sequences)

        return self.encoder(
            sequence_batch.sequences,
            attention_mask=(sequence_batch.attention_mask),
        )

    def forward(
        self,
        user: TensorizedUser,
    ) -> torch.Tensor:
        encoded_sequence = self.encode_sequence(user)

        return encoded_sequence[0]
