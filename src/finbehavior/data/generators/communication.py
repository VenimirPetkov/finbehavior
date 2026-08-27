import random
from datetime import datetime

from finbehavior.data.behavior_profile import BehaviorProfile
from finbehavior.data.reference.communication import (
    COMMUNICATION_CHANNELS,
    COMMUNICATION_ENGAGEMENTS,
    COMMUNICATION_TOPICS,
)
from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event


def _choose_engagement(
    behavior: BehaviorProfile,
    rng: random.Random,
) -> str:
    engagement = behavior.communication_engagement

    none_weight = 1.0 - (engagement * 0.70)
    opened_weight = engagement * 0.55
    clicked_weight = engagement * 0.15

    return rng.choices(
        population=COMMUNICATION_ENGAGEMENTS,
        weights=(
            none_weight,
            opened_weight,
            clicked_weight,
        ),
        k=1,
    )[0]


def generate_communication_event(
    behavior: BehaviorProfile,
    created: datetime,
    rng: random.Random | None = None,
) -> Event:
    rng = rng or random.Random()

    return Event(
        created=created,
        source=EventSource.COMMUNICATION,
        fields={
            "channel": rng.choice(COMMUNICATION_CHANNELS),
            "topic": rng.choice(COMMUNICATION_TOPICS),
            "engagement": _choose_engagement(
                behavior,
                rng,
            ),
        },
    )
