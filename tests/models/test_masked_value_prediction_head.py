import pytest
import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)


def test_masked_value_prediction_head_outputs_vocabulary_logits():
    vocabulary_size = 17

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=vocabulary_size,
    )

    representation = torch.randn(
        DEFAULT_EMBEDDING_DIMENSION,
    )

    logits = prediction_head(representation)

    assert logits.shape == (vocabulary_size,)


def test_masked_value_prediction_head_rejects_empty_vocabulary():
    with pytest.raises(
        ValueError,
        match="Vocabulary size must be positive",
    ):
        MaskedValuePredictionHead(
            vocabulary_size=0,
        )


def test_masked_value_prediction_head_masks_values_outside_key_domain():
    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=6,
        allowed_token_ids_by_key_id={
            4: (
                1,
                3,
            ),
        },
    )

    logits = prediction_head(
        torch.randn(DEFAULT_EMBEDDING_DIMENSION),
        key_ids=torch.tensor(4),
    )

    assert torch.isfinite(logits[1])
    assert torch.isfinite(logits[3])

    assert torch.isneginf(logits[0])
    assert torch.isneginf(logits[2])
    assert torch.isneginf(logits[4])
    assert torch.isneginf(logits[5])


def test_masked_value_prediction_head_requires_key_for_configured_domains():
    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=6,
        allowed_token_ids_by_key_id={
            4: (1,),
        },
    )

    with pytest.raises(
        ValueError,
        match="Key IDs are required",
    ):
        prediction_head(torch.randn(DEFAULT_EMBEDDING_DIMENSION))
