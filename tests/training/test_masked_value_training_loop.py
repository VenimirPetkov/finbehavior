import pytest
import torch

from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.training.masked_value_training_loop import (
    train_masked_values,
)


def test_training_loop_reduces_loss(
    masked_value_example_factory,
    trainable_stub_model,
):
    example = masked_value_example_factory(1)

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )

    with torch.no_grad():
        prediction_head.output_projection.weight.zero_()
        prediction_head.output_projection.bias.zero_()

    optimizer = torch.optim.SGD(
        list(trainable_stub_model.parameters())
        + list(prediction_head.parameters()),
        lr=0.1,
    )

    losses = train_masked_values(
        model=trainable_stub_model,
        prediction_head=prediction_head,
        optimizer=optimizer,
        examples=(example,),
        epoch_count=20,
    )

    assert len(losses) == 20

    assert losses[-1] < losses[0]


def test_training_loop_rejects_empty_examples(
    trainable_stub_model,
):
    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )

    optimizer = torch.optim.SGD(
        list(trainable_stub_model.parameters())
        + list(prediction_head.parameters()),
        lr=0.1,
    )

    with pytest.raises(
        ValueError,
        match="Training examples must not be empty",
    ):
        train_masked_values(
            model=trainable_stub_model,
            prediction_head=prediction_head,
            optimizer=optimizer,
            examples=(),
            epoch_count=1,
        )
