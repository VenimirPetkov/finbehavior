import pytest
import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.training.masked_value_batch_training_loss import (
    masked_value_batch_training_loss,
)
from finbehavior.training.masked_value_loss import (
    masked_value_loss,
)


class StubBatchFinBehaviorModel:
    def __init__(
        self,
        encoded_sequences: torch.Tensor,
    ) -> None:
        self.encoded_sequences = encoded_sequences

    def encode_users(
        self,
        users,
    ) -> torch.Tensor:
        return self.encoded_sequences


def test_batch_training_loss_uses_each_example_representation(
    masked_value_example_factory,
):
    first_example = masked_value_example_factory(10)

    second_example = masked_value_example_factory(11)

    encoded_sequences = torch.zeros(
        2,
        2,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    encoded_sequences[
        0,
        1,
        0,
    ] = 1.0

    encoded_sequences[
        1,
        1,
        1,
    ] = 1.0

    model = StubBatchFinBehaviorModel(
        encoded_sequences=encoded_sequences,
    )

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )

    loss = masked_value_batch_training_loss(
        model=model,
        prediction_head=prediction_head,
        examples=(
            first_example,
            second_example,
        ),
    )

    expected_representations = torch.stack(
        (
            encoded_sequences[0, 1],
            encoded_sequences[1, 1],
        )
    )

    expected_logits = prediction_head(expected_representations)

    expected_targets = torch.stack(
        (
            first_example.target_token_id,
            second_example.target_token_id,
        )
    )

    expected_loss = masked_value_loss(
        expected_logits,
        expected_targets,
    )

    assert loss.ndim == 0

    assert torch.allclose(
        loss,
        expected_loss,
    )


def test_batch_training_loss_rejects_empty_batch():
    encoded_sequences = torch.zeros(
        1,
        1,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    model = StubBatchFinBehaviorModel(
        encoded_sequences=encoded_sequences,
    )

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )

    with pytest.raises(
        ValueError,
        match="Batch examples must not be empty",
    ):
        masked_value_batch_training_loss(
            model=model,
            prediction_head=prediction_head,
            examples=(),
        )
