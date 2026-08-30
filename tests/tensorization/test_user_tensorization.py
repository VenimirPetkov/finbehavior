import torch

from finbehavior.tensorization.user import (
    tensorize_user,
)
from finbehavior.tokenization.types import (
    TokenizedEvent,
    TokenizedField,
    TokenizedProfile,
    TokenizedUser,
)


def test_tensorize_user():
    user = TokenizedUser(
        user_id=15,
        profile=TokenizedProfile(
            user_token_id=2,
            fields=(
                TokenizedField(
                    key_id=23,
                    value_id=113,
                ),
                TokenizedField(
                    key_id=24,
                    value_id=63,
                ),
            ),
        ),
        events=(
            TokenizedEvent(
                event_token_id=3,
                fields=(
                    TokenizedField(
                        key_id=6,
                        value_id=128,
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
            ),
            TokenizedEvent(
                event_token_id=3,
                fields=(
                    TokenizedField(
                        key_id=7,
                        value_id=32,
                    ),
                ),
                calendar_features=(
                    0.0,
                    1.0,
                    -0.5,
                    0.5,
                    0.25,
                    -0.25,
                ),
                elapsed_time_feature=8.0,
            ),
        ),
    )

    tensorized = tensorize_user(user)

    assert tensorized.user_id == 15

    assert torch.equal(
        tensorized.profile.key_ids,
        torch.tensor(
            [23, 24],
            dtype=torch.long,
        ),
    )

    assert torch.equal(
        tensorized.profile.value_ids,
        torch.tensor(
            [113, 63],
            dtype=torch.long,
        ),
    )

    assert len(tensorized.events) == 2

    assert tensorized.events[0].event_token_id.item() == 3
    assert tensorized.events[1].event_token_id.item() == 3

    assert torch.equal(
        tensorized.events[0].key_ids,
        torch.tensor(
            [6],
            dtype=torch.long,
        ),
    )

    assert torch.equal(
        tensorized.events[1].value_ids,
        torch.tensor(
            [32],
            dtype=torch.long,
        ),
    )
