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
from finbehavior.training.masked_value_training_loop import (
    train_masked_values,
)
from finbehavior.training.masking import (
    mask_event_value,
)


class TrainableStubFinBehaviorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        initial_representation = torch.zeros(
            DEFAULT_EMBEDDING_DIMENSION,
        )

        initial_representation[0] = 1.0

        self.event_representation = nn.Parameter(initial_representation)

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
            [1],
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


def test_training_loop_reduces_loss():
    example = build_example()

    model = TrainableStubFinBehaviorModel()

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )

    with torch.no_grad():
        prediction_head.output_projection.weight.zero_()
        prediction_head.output_projection.bias.zero_()

    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(prediction_head.parameters()),
        lr=0.1,
    )

    losses = train_masked_values(
        model=model,
        prediction_head=prediction_head,
        optimizer=optimizer,
        examples=(example,),
        epoch_count=20,
    )

    assert len(losses) == 20

    assert losses[-1] < losses[0]


def test_training_loop_rejects_empty_examples():
    model = TrainableStubFinBehaviorModel()

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )

    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(prediction_head.parameters()),
        lr=0.1,
    )

    with pytest.raises(
        ValueError,
        match="Training examples must not be empty",
    ):
        train_masked_values(
            model=model,
            prediction_head=prediction_head,
            optimizer=optimizer,
            examples=(),
            epoch_count=1,
        )
