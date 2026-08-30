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
from finbehavior.training.masked_value_training_step import (
    masked_value_training_step,
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


def test_masked_value_training_step_updates_weights():
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

    example = mask_event_value(
        user=user,
        event_index=0,
        field_index=0,
        mask_token_id=0,
    )

    model = TrainableStubFinBehaviorModel()

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
        list(model.parameters()) + list(prediction_head.parameters()),
        lr=0.1,
    )

    model_before = model.event_representation.detach().clone()

    head_before = prediction_head.output_projection.weight.detach().clone()

    loss = masked_value_training_step(
        model=model,
        prediction_head=prediction_head,
        optimizer=optimizer,
        example=example,
    )

    assert loss.ndim == 0

    assert not torch.allclose(
        model.event_representation,
        model_before,
    )

    assert not torch.allclose(
        prediction_head.output_projection.weight,
        head_before,
    )
