import random
from datetime import datetime

from finbehavior.data.generators.trading import (
    generate_trading_event,
)
from finbehavior.data.reference.trading import (
    TRADING_ACTIONS,
    TRADING_CURRENCIES,
    TRADING_INSTRUMENTS,
)
from finbehavior.domain.enums import EventSource


def test_generate_trading_event():
    created = datetime(2026, 8, 27, 15, 30)

    event = generate_trading_event(
        created=created,
        rng=random.Random(42),
    )

    assert event.created == created
    assert event.source == EventSource.TRADING

    assert event.fields["action"] in TRADING_ACTIONS

    assert event.fields["asset_class"] in TRADING_INSTRUMENTS

    asset_class = event.fields["asset_class"]

    assert event.fields["instrument"] in TRADING_INSTRUMENTS[asset_class]

    assert event.fields["currency"] in TRADING_CURRENCIES

    assert event.fields["amount"] > 0


def test_generate_trading_event_is_reproducible():
    created = datetime(2026, 8, 27, 15, 30)

    first = generate_trading_event(
        created=created,
        rng=random.Random(42),
    )

    second = generate_trading_event(
        created=created,
        rng=random.Random(42),
    )

    assert first == second
