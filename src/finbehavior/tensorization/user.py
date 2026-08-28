from finbehavior.tokenization.types import (
    TokenizedUser,
)

from .event import tensorize_event
from .profile import tensorize_profile
from .types import TensorizedUser


def tensorize_user(
    user: TokenizedUser,
) -> TensorizedUser:
    profile = tensorize_profile(user.profile)

    events = tuple(tensorize_event(event) for event in user.events)

    return TensorizedUser(
        user_id=user.user_id,
        profile=profile,
        events=events,
    )
