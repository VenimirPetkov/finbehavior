import random
from datetime import datetime

from finbehavior.data.generators.app import generate_app_event
from finbehavior.data.reference.app import (
    APP_ACTIONS,
    APP_SCREENS,
)
from finbehavior.domain.enums import EventSource


def test_generate_app_event():
    created = datetime(2026, 8, 27, 10, 30)

    event = generate_app_event(
        created=created,
        rng=random.Random(42),
    )

    assert event.created == created
    assert event.source == EventSource.APP
    assert event.fields["screen"] in APP_SCREENS
    assert event.fields["action"] in APP_ACTIONS


def test_generate_app_event_is_reproducible():
    created = datetime(2026, 8, 27, 10, 30)

    first = generate_app_event(
        created=created,
        rng=random.Random(42),
    )

    second = generate_app_event(
        created=created,
        rng=random.Random(42),
    )

    assert first == second
