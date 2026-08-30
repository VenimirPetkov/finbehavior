import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.self_attention import (
    SelfAttention,
)


def test_self_attention_preserves_sequence_shape():
    sequence_length = 4

    attention = SelfAttention()

    sequence = torch.randn(
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    output = attention(sequence)

    assert output.shape == (
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )


def test_self_attention_has_learnable_projections():
    attention = SelfAttention()

    assert (
        attention.query_projection.weight.requires_grad
        is True
    )

    assert (
        attention.key_projection.weight.requires_grad
        is True
    )

    assert (
        attention.value_projection.weight.requires_grad
        is True
    )