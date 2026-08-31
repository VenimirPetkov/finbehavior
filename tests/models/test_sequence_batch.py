import pytest
import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.sequence_batch import (
    build_sequence_batch,
)


def test_build_sequence_batch_pads_different_lengths():
    first_sequence = torch.randn(
        3,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    second_sequence = torch.randn(
        5,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    third_sequence = torch.randn(
        2,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    batch = build_sequence_batch(
        (
            first_sequence,
            second_sequence,
            third_sequence,
        )
    )

    assert batch.sequences.shape == (
        3,
        5,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    expected_mask = torch.tensor(
        [
            [
                True,
                True,
                True,
                False,
                False,
            ],
            [
                True,
                True,
                True,
                True,
                True,
            ],
            [
                True,
                True,
                False,
                False,
                False,
            ],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(
        batch.attention_mask,
        expected_mask,
    )


def test_build_sequence_batch_rejects_empty_batch():
    with pytest.raises(
        ValueError,
        match=("Sequence batch must contain " "at least one sequence"),
    ):
        build_sequence_batch(())
