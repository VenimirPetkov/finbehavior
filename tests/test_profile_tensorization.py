import torch

from finbehavior.tensorization.profile import (
    tensorize_profile,
)
from finbehavior.tokenization.types import (
    TokenizedField,
    TokenizedProfile,
)


def test_tensorize_profile():
    profile = TokenizedProfile(
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
            TokenizedField(
                key_id=25,
                value_id=123,
            ),
        ),
    )

    tensorized = tensorize_profile(profile)

    assert torch.equal(
        tensorized.key_ids,
        torch.tensor(
            [23, 24, 25],
            dtype=torch.long,
        ),
    )

    assert torch.equal(
        tensorized.value_ids,
        torch.tensor(
            [113, 63, 123],
            dtype=torch.long,
        ),
    )

    assert tensorized.user_token_id.item() == 2
