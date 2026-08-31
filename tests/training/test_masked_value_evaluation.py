import pytest
import torch
from torch import nn

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.tensorization.types import (
    TensorizedUser,
)
from finbehavior.training.masked_value_evaluation import (
    evaluate_masked_values,
)


class EvaluationStubModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.event_representation = nn.Parameter(
            torch.ones(
                DEFAULT_EMBEDDING_DIMENSION,
            )
        )

    def encode_sequence(
        self,
        user: TensorizedUser,
    ) -> torch.Tensor:
        profile_representation = torch.zeros_like(self.event_representation)

        return torch.stack(
            (
                profile_representation,
                self.event_representation,
            ),
            dim=0,
        )

    def encode_users(
        self,
        users: tuple[TensorizedUser, ...],
    ) -> torch.Tensor:
        return torch.stack(
            tuple(self.encode_sequence(user) for user in users),
            dim=0,
        )


def test_evaluate_masked_values_returns_average_loss(
    masked_value_example_factory,
):
    model = EvaluationStubModel()

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )

    example = masked_value_example_factory(10)

    loss = evaluate_masked_values(
        model=model,
        prediction_head=prediction_head,
        examples=(
            example,
            example,
        ),
    )

    assert isinstance(
        loss,
        float,
    )

    assert loss > 0.0


def test_evaluation_does_not_change_weights(
    masked_value_example_factory,
):
    model = EvaluationStubModel()

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )

    example = masked_value_example_factory(10)

    model_before = model.event_representation.detach().clone()

    head_before = prediction_head.output_projection.weight.detach().clone()

    evaluate_masked_values(
        model=model,
        prediction_head=prediction_head,
        examples=(example,),
    )

    assert torch.equal(
        model.event_representation,
        model_before,
    )

    assert torch.equal(
        prediction_head.output_projection.weight,
        head_before,
    )


def test_evaluation_restores_training_mode(
    masked_value_example_factory,
):
    model = EvaluationStubModel()
    model.train()

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )
    prediction_head.train()

    example = masked_value_example_factory(10)

    evaluate_masked_values(
        model=model,
        prediction_head=prediction_head,
        examples=(example,),
    )

    assert model.training
    assert prediction_head.training


def test_evaluation_rejects_empty_examples():
    model = EvaluationStubModel()

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )

    with pytest.raises(
        ValueError,
        match="Evaluation examples must not be empty",
    ):
        evaluate_masked_values(
            model=model,
            prediction_head=prediction_head,
            examples=(),
        )
