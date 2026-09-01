import pytest
import torch
from torch import nn

from finbehavior.evaluation.masked_value_metrics import (
    evaluate_masked_value_metrics,
)
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
from finbehavior.training.masking import (
    mask_event_value,
)


class MetricsStubModel(nn.Module):
    def __init__(
        self,
        encoded_sequences: torch.Tensor,
    ) -> None:
        super().__init__()

        self.dummy_parameter = nn.Parameter(torch.zeros(1))

        self.encoded_sequences = encoded_sequences

    def encode_users(
        self,
        users: tuple[TensorizedUser, ...],
    ) -> torch.Tensor:
        return self.encoded_sequences.to(self.dummy_parameter.device)


def build_example(
    user_id: int,
    target_token_id: int,
):
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
            [target_token_id],
            dtype=torch.long,
        ),
        calendar_features=torch.zeros(
            CALENDAR_FEATURE_DIMENSION,
            dtype=torch.float32,
        ),
        elapsed_time_feature=torch.tensor(
            0.0,
            dtype=torch.float32,
        ),
    )

    user = TensorizedUser(
        user_id=user_id,
        profile=profile,
        events=(event,),
    )

    return mask_event_value(
        user=user,
        event_index=0,
        field_index=0,
        mask_token_id=5,
    )


def test_evaluate_masked_value_metrics():
    examples = (
        build_example(1, 1),
        build_example(2, 0),
        build_example(3, 2),
        build_example(4, 1),
    )

    encoded_sequences = torch.zeros(
        4,
        2,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    encoded_sequences[0, 1, 0] = 1.0
    encoded_sequences[1, 1, 1] = 1.0
    encoded_sequences[2, 1, 2] = 1.0
    encoded_sequences[3, 1, 3] = 1.0

    model = MetricsStubModel(
        encoded_sequences=encoded_sequences,
    )

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=6,
    )

    with torch.no_grad():
        prediction_head.output_projection.weight.zero_()
        prediction_head.output_projection.bias.zero_()

        prediction_head.output_projection.weight[
            1,
            0,
        ] = 5.0

        prediction_head.output_projection.weight[
            0,
            1,
        ] = 5.0

        prediction_head.output_projection.weight[
            2,
            2,
        ] = 5.0

        prediction_head.output_projection.weight[
            0,
            3,
        ] = 5.0

        prediction_head.output_projection.weight[
            1,
            3,
        ] = 4.0

    metrics = evaluate_masked_value_metrics(
        model=model,
        prediction_head=prediction_head,
        examples=examples,
        batch_size=4,
    )

    assert metrics.loss > 0.0

    assert metrics.top_1_accuracy == pytest.approx(0.75)

    assert metrics.top_5_accuracy == pytest.approx(1.0)
