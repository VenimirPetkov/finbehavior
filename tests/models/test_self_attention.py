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


def test_self_attention_preserves_batch_shape():
    batch_size = 3
    sequence_length = 5

    attention = SelfAttention()

    batch = torch.randn(
        batch_size,
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    output = attention(batch)

    assert output.shape == (
        batch_size,
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )


def test_self_attention_ignores_masked_key_positions():
    sequence_length = 4

    attention = SelfAttention()

    sequence = torch.randn(
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    attention_mask = torch.tensor(
        [
            True,
            True,
            True,
            False,
        ],
        dtype=torch.bool,
    )

    changed_sequence = sequence.clone()

    changed_sequence[-1] = (
        torch.randn(
            DEFAULT_EMBEDDING_DIMENSION,
        )
        * 1000.0
    )

    original_output = attention(
        sequence,
        attention_mask=attention_mask,
    )

    changed_output = attention(
        changed_sequence,
        attention_mask=attention_mask,
    )

    assert torch.allclose(
        original_output[:-1],
        changed_output[:-1],
        atol=1e-5,
    )


def test_self_attention_has_learnable_projections():
    attention = SelfAttention()

    assert attention.query_projection.weight.requires_grad is True

    assert attention.key_projection.weight.requires_grad is True

    assert attention.value_projection.weight.requires_grad is True
