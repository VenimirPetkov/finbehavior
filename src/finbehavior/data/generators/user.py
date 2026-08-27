import random
from datetime import datetime

from finbehavior.data.generators.behavior_profile import (
    generate_behavior_profile,
)
from finbehavior.data.generators.history import (
    generate_event_history,
)
from finbehavior.data.generators.profile import (
    generate_profile_state,
)
from finbehavior.data.synthetic_user import SyntheticUser
from finbehavior.domain.record import UserRecord


def generate_synthetic_user(
    user_id: int,
    start: datetime,
    evaluation_point: datetime,
    rng: random.Random | None = None,
) -> SyntheticUser:
    if rng is None:
        rng = random.Random()

    behavior = generate_behavior_profile(rng)

    profile = generate_profile_state(
        behavior=behavior,
        rng=rng,
    )

    events = generate_event_history(
        behavior=behavior,
        profile=profile,
        start=start,
        end=evaluation_point,
        rng=rng,
    )

    record = UserRecord(
        user_id=user_id,
        evaluation_point=evaluation_point,
        profile=profile,
        events=events,
    )

    return SyntheticUser(
        behavior=behavior,
        record=record,
    )
