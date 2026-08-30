import torch

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


def test_mask_event_value_replaces_value_and_preserves_target():
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
            [10, 11],
            dtype=torch.long,
        ),
        value_ids=torch.tensor(
            [20, 21],
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

    mask_token_id = 1

    example = mask_event_value(
        user=user,
        event_index=0,
        field_index=1,
        mask_token_id=mask_token_id,
    )

    assert example.target_token_id.item() == 21

    assert example.user.events[0].value_ids.tolist() == [20, mask_token_id]

    assert user.events[0].value_ids.tolist() == [
        20,
        21,
    ]

    assert example.event_index == 0
    assert example.field_index == 1
