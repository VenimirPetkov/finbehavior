import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.self_attention import (
    SelfAttention,
)
from finbehavior.models.self_attention_block import (
    SelfAttentionBlock,
)


def test_self_attention_block_preserves_sequence_shape():
    sequence_length = 4

    self_attention = SelfAttention()

    block = SelfAttentionBlock(
        self_attention=self_attention,
    )

    sequence = torch.randn(
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    output = block(sequence)

    assert output.shape == (
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )


def test_self_attention_block_has_residual_connection():
    self_attention = SelfAttention()

    torch.nn.init.zeros_(self_attention.query_projection.weight)
    torch.nn.init.zeros_(self_attention.key_projection.weight)
    torch.nn.init.zeros_(self_attention.value_projection.weight)

    block = SelfAttentionBlock(
        self_attention=self_attention,
    )

    sequence = torch.randn(
        4,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    output = block(sequence)

    expected = block.layer_norm(sequence)

    assert torch.allclose(
        output,
        expected,
    )
