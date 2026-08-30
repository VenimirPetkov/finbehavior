import torch

from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.training.masked_value_training_step import (
    masked_value_training_step,
)


def test_masked_value_training_step_updates_weights(
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

        prediction_head.output_projection.weight[
            0,
            0,
        ] = 1.0

    optimizer = torch.optim.SGD(
        list(trainable_stub_model.parameters()) + list(prediction_head.parameters()),
        lr=0.1,
    )

    model_before = trainable_stub_model.event_representation.detach().clone()

    head_before = prediction_head.output_projection.weight.detach().clone()

    loss = masked_value_training_step(
        model=trainable_stub_model,
        prediction_head=prediction_head,
        optimizer=optimizer,
        example=example,
    )

    assert loss.ndim == 0

    assert not torch.allclose(
        trainable_stub_model.event_representation,
        model_before,
    )

    assert not torch.allclose(
        prediction_head.output_projection.weight,
        head_before,
    )
