import pytest
import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.encoder import (
    TransformerEncoder,
)


def test_transformer_encoder_preserves_sequence_shape(
    transformer_block_factory,
):
    layer_count = 3

    blocks = tuple(transformer_block_factory() for _ in range(layer_count))

    encoder = TransformerEncoder(
        blocks=blocks,
    )

    sequence_length = 4

    sequence = torch.randn(
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    output = encoder(sequence)

    assert output.shape == (
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    assert len(encoder.blocks) == layer_count


def test_transformer_encoder_rejects_empty_blocks():
    with pytest.raises(
        ValueError,
        match="Encoder must contain at least one transformer block",
    ):
        TransformerEncoder(
            blocks=(),
        )
