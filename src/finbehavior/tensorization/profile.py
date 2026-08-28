import torch

from finbehavior.tokenization.types import (
    TokenizedProfile,
)

from .types import TensorizedProfile


def tensorize_profile(
    profile: TokenizedProfile,
) -> TensorizedProfile:
    key_ids = torch.tensor(
        [field.key_id for field in profile.fields],
        dtype=torch.long,
    )

    value_ids = torch.tensor(
        [field.value_id for field in profile.fields],
        dtype=torch.long,
    )

    user_token_id = torch.tensor(
        profile.user_token_id,
        dtype=torch.long,
    )

    return TensorizedProfile(
        user_token_id=user_token_id,
        key_ids=key_ids,
        value_ids=value_ids,
    )
