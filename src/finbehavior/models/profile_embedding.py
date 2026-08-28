import torch
from torch import nn

from finbehavior.tensorization.types import (
    TensorizedProfile,
)

from .field_embedding import FieldEmbedding


class ProfileEmbedding(nn.Module):
    def __init__(
        self,
        field_embedding: FieldEmbedding,
    ) -> None:
        super().__init__()

        self.field_embedding = field_embedding

    def forward(
        self,
        profile: TensorizedProfile,
    ) -> torch.Tensor:
        user_vector = self.field_embedding.token_embedding(profile.user_token_id)

        if profile.key_ids.numel() == 0:
            return user_vector

        field_vectors = self.field_embedding(
            key_ids=profile.key_ids,
            value_ids=profile.value_ids,
        )

        fields_vector = field_vectors.mean(dim=0)

        return user_vector + fields_vector
