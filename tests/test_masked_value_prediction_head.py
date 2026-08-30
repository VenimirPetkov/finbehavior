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
