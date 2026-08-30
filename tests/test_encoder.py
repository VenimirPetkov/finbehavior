import pytest
import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.encoder import (
    TransformerEncoder,
)
from finbehavior.models.feed_forward import (
    FeedForward,
)
from finbehavior.models.self_attention import (
    SelfAttention,
)
from finbehavior.models.self_attention_block import (
    SelfAttentionBlock,
)
from finbehavior.models.transformer_block import (
    TransformerBlock,
)


def build_transformer_block():
    self_attention = SelfAttention()

    self_attention_block = SelfAttentionBlock(
        self_attention=self_attention,
    )

    feed_forward = FeedForward()

    return TransformerBlock(
        self_attention_block=self_attention_block,
        feed_forward=feed_forward,
    )


def test_transformer_encoder_preserves_sequence_shape():
    layer_count = 3

    blocks = tuple(build_transformer_block() for _ in range(layer_count))

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
