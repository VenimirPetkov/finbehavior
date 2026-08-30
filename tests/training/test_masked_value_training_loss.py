import torch

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
from finbehavior.training.masked_value_loss import (
    masked_value_loss,
)
from finbehavior.training.masked_value_training_loss import (
    masked_value_training_loss,
)
from finbehavior.training.masking import (
    mask_event_value,
)


class StubFinBehaviorModel:
    def __init__(
        self,
        encoded_sequence: torch.Tensor,
    ) -> None:
        self.encoded_sequence = encoded_sequence

    def encode_sequence(
        self,
        user: TensorizedUser,
    ) -> torch.Tensor:
        return self.encoded_sequence


def build_event(
    value_id: int,
) -> TensorizedEvent:
    return TensorizedEvent(
        event_token_id=torch.tensor(
            3,
            dtype=torch.long,
        ),
        key_ids=torch.tensor(
            [10],
            dtype=torch.long,
        ),
        value_ids=torch.tensor(
            [value_id],
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


def test_masked_value_training_loss_uses_masked_event_representation():
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

    user = TensorizedUser(
        user_id=1,
        profile=profile,
        events=(
            build_event(0),
            build_event(2),
        ),
    )

    example = mask_event_value(
        user=user,
        event_index=1,
        field_index=0,
        mask_token_id=1,
    )

    encoded_sequence = torch.zeros(
        3,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    encoded_sequence[1, 0] = 1.0
    encoded_sequence[2, 1] = 1.0

    model = StubFinBehaviorModel(
        encoded_sequence=encoded_sequence,
    )

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=4,
    )

    with torch.no_grad():
        prediction_head.output_projection.weight.zero_()
        prediction_head.output_projection.bias.zero_()

        prediction_head.output_projection.weight[
            2,
            1,
        ] = 5.0

    loss = masked_value_training_loss(
        model=model,
        prediction_head=prediction_head,
        example=example,
    )

    expected_logits = prediction_head(encoded_sequence[2])

    expected_loss = masked_value_loss(
        expected_logits,
        example.target_token_id,
    )

    assert loss.ndim == 0

    assert torch.allclose(
        loss,
        expected_loss,
    )
