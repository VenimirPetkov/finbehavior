from collections.abc import Mapping, Sequence

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
        allowed_token_ids_by_key_id: (
            Mapping[
                int,
                Sequence[int],
            ]
            | None
        ) = None,
    ) -> None:
        super().__init__()

        if vocabulary_size <= 0:
            raise ValueError("Vocabulary size must be positive")

        self.output_projection = nn.Linear(
            embedding_dimension,
            vocabulary_size,
        )

        if allowed_token_ids_by_key_id is None:
            allowed_value_mask = torch.empty(
                0,
                dtype=torch.bool,
            )
        else:
            allowed_value_mask = _build_allowed_value_mask(
                vocabulary_size=vocabulary_size,
                allowed_token_ids_by_key_id=(allowed_token_ids_by_key_id),
            )

        self.register_buffer(
            "allowed_value_mask",
            allowed_value_mask,
            persistent=False,
        )

    def forward(
        self,
        representation: torch.Tensor,
        key_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.output_projection(representation)

        if self.allowed_value_mask.numel() == 0:
            return logits

        if key_ids is None:
            raise ValueError(
                "Key IDs are required when value-domain masking is configured"
            )

        if key_ids.shape != logits.shape[:-1]:
            raise ValueError(
                "Key IDs must match the leading dimensions of the representations"
            )

        allowed = self.allowed_value_mask[key_ids]

        if not torch.all(allowed.any(dim=-1)):
            raise ValueError(
                "No allowed value domain configured for one or more key IDs"
            )

        return logits.masked_fill(
            ~allowed,
            float("-inf"),
        )

    def get_allowed_token_ids_by_key_id(
        self,
    ) -> dict[int, tuple[int, ...]] | None:
        if self.allowed_value_mask.numel() == 0:
            return None

        domains = {}

        for key_id, allowed in enumerate(self.allowed_value_mask):
            token_ids = tuple(
                int(token_id)
                for token_id in torch.nonzero(
                    allowed,
                    as_tuple=False,
                ).flatten()
            )

            if token_ids:
                domains[key_id] = token_ids

        return domains


def _build_allowed_value_mask(
    vocabulary_size: int,
    allowed_token_ids_by_key_id: Mapping[
        int,
        Sequence[int],
    ],
) -> torch.Tensor:
    mask = torch.zeros(
        vocabulary_size,
        vocabulary_size,
        dtype=torch.bool,
    )

    for key_id, allowed_token_ids in allowed_token_ids_by_key_id.items():
        if key_id < 0 or key_id >= vocabulary_size:
            raise ValueError(f"Key ID is outside the vocabulary: {key_id}")

        allowed_token_ids = tuple(allowed_token_ids)

        if not allowed_token_ids:
            raise ValueError(
                f"Allowed value domain must not be empty for key ID {key_id}"
            )

        if any(
            token_id < 0 or token_id >= vocabulary_size
            for token_id in allowed_token_ids
        ):
            raise ValueError(
                f"Allowed token ID is outside the vocabulary for key ID {key_id}"
            )

        mask[
            key_id,
            torch.tensor(
                allowed_token_ids,
                dtype=torch.long,
            ),
        ] = True

    return mask
