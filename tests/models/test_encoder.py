import pytest
import torch

from finbehavior.models.config.embedding import DEFAULT_EMBEDDING_DIMENSION
from finbehavior.models.encoder import TransformerEncoder


def test_transformer_encoder_accepts_attention_mask(
    transformer_block_factory,
):
    blocks = (
        transformer_block_factory(),
        transformer_block_factory(),
    )

    encoder = TransformerEncoder(
        blocks=blocks,
    )

    batch_size = 2
    sequence_length = 4

    sequence = torch.randn(
        batch_size,
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    attention_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )

    output = encoder(
        sequence,
        attention_mask=attention_mask,
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )


def test_transformer_encoder_rejects_empty_blocks():
    with pytest.raises(
        ValueError,
        match="Encoder must contain at least one transformer block",
    ):
        TransformerEncoder(
            blocks=(),
        )
