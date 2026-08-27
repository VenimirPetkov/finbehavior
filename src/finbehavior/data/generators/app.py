import random
from datetime import datetime

from finbehavior.data.reference.app import (
    APP_ACTIONS,
    APP_SCREENS,
)
from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event


def generate_app_event(
    created: datetime,
    rng: random.Random | None = None,
) -> Event:
    rng = rng or random.Random()

    return Event(
        created=created,
        source=EventSource.APP,
        fields={
            "screen": rng.choice(APP_SCREENS),
            "action": rng.choice(APP_ACTIONS),
        },
    )
