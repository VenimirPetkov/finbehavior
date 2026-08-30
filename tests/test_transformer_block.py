import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
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

    transformer_block = TransformerBlock(
        self_attention_block=self_attention_block,
        feed_forward=feed_forward,
    )

    return transformer_block


def test_transformer_block_preserves_sequence_shape():
    transformer_block = build_transformer_block()

    sequence_length = 4

    sequence = torch.randn(
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    output = transformer_block(sequence)

    assert output.shape == (
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )


def test_transformer_block_has_feed_forward_residual():
    transformer_block = build_transformer_block()

    torch.nn.init.zeros_(transformer_block.feed_forward.output_projection.weight)

    torch.nn.init.zeros_(transformer_block.feed_forward.output_projection.bias)

    sequence = torch.randn(
        4,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    attention_output = transformer_block.self_attention_block(sequence)

    output = transformer_block(sequence)

    expected = transformer_block.feed_forward_layer_norm(attention_output)

    assert torch.allclose(
        output,
        expected,
    )
