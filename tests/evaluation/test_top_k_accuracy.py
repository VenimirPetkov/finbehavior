import pytest
import torch

from finbehavior.evaluation.top_k_accuracy import (
    top_k_accuracy,
)


def test_top_1_accuracy():
    logits = torch.tensor(
        [
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            1,
            0,
            2,
            1,
        ],
        dtype=torch.long,
    )

    accuracy = top_k_accuracy(
        logits=logits,
        target_token_ids=targets,
        k=1,
    )

    assert accuracy == pytest.approx(0.75)


def test_top_2_accuracy():
    logits = torch.tensor(
        [
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            1,
            0,
            2,
            1,
        ],
        dtype=torch.long,
    )

    accuracy = top_k_accuracy(
        logits=logits,
        target_token_ids=targets,
        k=2,
    )

    assert accuracy == pytest.approx(1.0)


def test_top_k_accuracy_rejects_invalid_k():
    logits = torch.zeros(
        2,
        4,
    )

    targets = torch.zeros(
        2,
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="k must be between 1 and vocabulary size",
    ):
        top_k_accuracy(
            logits=logits,
            target_token_ids=targets,
            k=5,
        )
