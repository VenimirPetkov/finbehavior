import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.config.feed_forward import (
    DEFAULT_FEED_FORWARD_EXPANSION_FACTOR,
)
from finbehavior.models.feed_forward import (
    FeedForward,
)


def test_feed_forward_preserves_sequence_shape():
    sequence_length = 4

    feed_forward = FeedForward()

    sequence = torch.randn(
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    output = feed_forward(sequence)

    assert output.shape == (
        sequence_length,
        DEFAULT_EMBEDDING_DIMENSION,
    )


def test_feed_forward_expands_hidden_dimension():
    feed_forward = FeedForward()

    expected_hidden_dimension = (
        DEFAULT_EMBEDDING_DIMENSION * DEFAULT_FEED_FORWARD_EXPANSION_FACTOR
    )

    assert feed_forward.input_projection.out_features == expected_hidden_dimension

    assert feed_forward.output_projection.in_features == expected_hidden_dimension
