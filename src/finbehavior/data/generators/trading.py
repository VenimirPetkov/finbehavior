import random
from datetime import datetime

from finbehavior.data.reference.trading import (
    TRADING_ACTIONS,
    TRADING_CURRENCIES,
    TRADING_INSTRUMENTS,
)
from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event


def generate_trading_event(
    created: datetime,
    rng: random.Random | None = None,
) -> Event:
    rng = rng or random.Random()

    asset_class = rng.choice(tuple(TRADING_INSTRUMENTS.keys()))

    instrument = rng.choice(TRADING_INSTRUMENTS[asset_class])

    action = rng.choice(TRADING_ACTIONS)

    currency = rng.choice(TRADING_CURRENCIES)

    amount = round(
        rng.uniform(10.0, 2000.0),
        2,
    )

    return Event(
        created=created,
        source=EventSource.TRADING,
        fields={
            "action": action,
            "asset_class": asset_class,
            "instrument": instrument,
            "amount": amount,
            "currency": currency,
        },
    )
