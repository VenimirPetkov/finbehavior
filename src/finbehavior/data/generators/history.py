import random
from datetime import datetime

from finbehavior.data.behavior_profile import BehaviorProfile
from finbehavior.data.generators.app import generate_app_event
from finbehavior.data.generators.communication import (
    generate_communication_event,
)
from finbehavior.data.generators.timeline import (
    generate_event_timestamps,
)
from finbehavior.data.generators.trading import (
    generate_trading_event,
)
from finbehavior.data.generators.transaction import (
    generate_transaction_event,
)
from finbehavior.data.reference.event_rates import (
    APP_ACTIVITY_RATE,
    APP_BASE_RATE,
    COMMUNICATION_BASE_RATE,
    TRADING_ACTIVITY_RATE,
    TRADING_BASE_RATE,
    TRANSACTION_BASE_RATE,
    TRANSACTION_SPENDING_RATE,
)
from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event
from finbehavior.domain.profile import ProfileState


def calculate_daily_event_rates(
    behavior: BehaviorProfile,
) -> dict[EventSource, float]:
    return {
        EventSource.TRANSACTION: (
            TRANSACTION_BASE_RATE
            + behavior.spending_tendency * TRANSACTION_SPENDING_RATE
        ),
        EventSource.APP: (
            APP_BASE_RATE
            + behavior.app_activity * APP_ACTIVITY_RATE
        ),
        EventSource.TRADING: (
            TRADING_BASE_RATE
            + behavior.investing_tendency * TRADING_ACTIVITY_RATE
        ),
        EventSource.COMMUNICATION: COMMUNICATION_BASE_RATE,
    }


def generate_event_history(
    behavior: BehaviorProfile,
    profile: ProfileState,
    start: datetime,
    end: datetime,
    rng: random.Random | None = None,
) -> list[Event]:
    rng = rng or random.Random()

    region = profile.fields.get("region")

    if not isinstance(region, str):
        raise ValueError("Profile must contain a valid region")

    rates = calculate_daily_event_rates(behavior)

    events: list[Event] = []

    transaction_timestamps = generate_event_timestamps(
        start=start,
        end=end,
        daily_rate=rates[EventSource.TRANSACTION],
        rng=rng,
    )

    for created in transaction_timestamps:
        events.append(
            generate_transaction_event(
                behavior=behavior,
                created=created,
                home_region=region,
                rng=rng,
            )
        )

    app_timestamps = generate_event_timestamps(
        start=start,
        end=end,
        daily_rate=rates[EventSource.APP],
        rng=rng,
    )

    for created in app_timestamps:
        events.append(
            generate_app_event(
                created=created,
                rng=rng,
            )
        )

    trading_timestamps = generate_event_timestamps(
        start=start,
        end=end,
        daily_rate=rates[EventSource.TRADING],
        rng=rng,
    )

    for created in trading_timestamps:
        events.append(
            generate_trading_event(
                created=created,
                rng=rng,
            )
        )

    communication_timestamps = generate_event_timestamps(
        start=start,
        end=end,
        daily_rate=rates[EventSource.COMMUNICATION],
        rng=rng,
    )

    for created in communication_timestamps:
        events.append(
            generate_communication_event(
                behavior=behavior,
                created=created,
                rng=rng,
            )
        )

    events.sort(key=lambda event: event.created)

    return events