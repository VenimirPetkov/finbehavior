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
    TensorizedEvent,
    TensorizedProfile,
    TensorizedUser,
)
from finbehavior.tokenization.config.temporal import (
    CALENDAR_FEATURE_DIMENSION,
)
from finbehavior.training.masked_value_evaluation import (
    evaluate_masked_values,
)
from finbehavior.training.masking import (
    mask_event_value,
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


def build_example():
    profile = TensorizedProfile(
        user_token_id=torch.tensor(
            2,
            dtype=torch.long,
        ),
        key_ids=torch.empty(
            0,
            dtype=torch.long,
        ),
        value_ids=torch.empty(
            0,
            dtype=torch.long,
        ),
    )

    event = TensorizedEvent(
        event_token_id=torch.tensor(
            3,
            dtype=torch.long,
        ),
        key_ids=torch.tensor(
            [10],
            dtype=torch.long,
        ),
        value_ids=torch.tensor(
            [2],
            dtype=torch.long,
        ),
        calendar_features=torch.zeros(
            CALENDAR_FEATURE_DIMENSION,
            dtype=torch.float32,
        ),
        elapsed_time_feature=torch.tensor(
            0.5,
            dtype=torch.float32,
        ),
    )

    user = TensorizedUser(
        user_id=1,
        profile=profile,
        events=(event,),
    )

    return mask_event_value(
        user=user,
        event_index=0,
        field_index=0,
        mask_token_id=0,
    )


def test_evaluate_masked_values_returns_average_loss():
    model = EvaluationStubModel()

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )

    example = build_example()

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


def test_evaluation_does_not_change_weights():
    model = EvaluationStubModel()

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )

    example = build_example()

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


def test_evaluation_restores_training_mode():
    model = EvaluationStubModel()
    model.train()

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )
    prediction_head.train()

    evaluate_masked_values(
        model=model,
        prediction_head=prediction_head,
        examples=(build_example(),),
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
