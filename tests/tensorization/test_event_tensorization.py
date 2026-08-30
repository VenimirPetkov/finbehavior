import torch

from finbehavior.tensorization.event import (
    tensorize_event,
)
from finbehavior.tokenization.types import (
    TokenizedEvent,
    TokenizedField,
)


def test_tensorize_event():
    event = TokenizedEvent(
        event_token_id=3,
        fields=(
            TokenizedField(
                key_id=6,
                value_id=128,
            ),
            TokenizedField(
                key_id=7,
                value_id=32,
            ),
        ),
        calendar_features=(
            0.5,
            -0.5,
            0.25,
            -0.25,
            1.0,
            0.0,
        ),
        elapsed_time_feature=17.42,
    )

    tensorized = tensorize_event(event)

    assert tensorized.event_token_id.item() == 3

    assert torch.equal(
        tensorized.key_ids,
        torch.tensor(
            [6, 7],
            dtype=torch.long,
        ),
    )

    assert torch.equal(
        tensorized.value_ids,
        torch.tensor(
            [128, 32],
            dtype=torch.long,
        ),
    )

    assert torch.equal(
        tensorized.calendar_features,
        torch.tensor(
            [
                0.5,
                -0.5,
                0.25,
                -0.25,
                1.0,
                0.0,
            ],
            dtype=torch.float32,
        ),
    )

    assert torch.equal(
        tensorized.elapsed_time_feature,
        torch.tensor(
            17.42,
            dtype=torch.float32,
        ),
    )
