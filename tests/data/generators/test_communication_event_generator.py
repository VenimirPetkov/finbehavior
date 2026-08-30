import random
from datetime import datetime

from finbehavior.data.behavior_profile import BehaviorProfile
from finbehavior.data.generators.communication import (
    generate_communication_event,
)
from finbehavior.data.reference.communication import (
    COMMUNICATION_CHANNELS,
    COMMUNICATION_ENGAGEMENTS,
    COMMUNICATION_TOPICS,
)
from finbehavior.domain.enums import EventSource


def create_behavior(
    communication_engagement: float = 0.6,
) -> BehaviorProfile:
    return BehaviorProfile(
        income_level=0.7,
        spending_tendency=0.6,
        travel_tendency=0.4,
        investing_tendency=0.5,
        app_activity=0.8,
        communication_engagement=communication_engagement,
    )


def test_generate_communication_event():
    created = datetime(2026, 8, 27, 12, 0)

    event = generate_communication_event(
        behavior=create_behavior(),
        created=created,
        rng=random.Random(42),
    )

    assert event.created == created
    assert event.source == EventSource.COMMUNICATION

    assert event.fields["channel"] in COMMUNICATION_CHANNELS
    assert event.fields["topic"] in COMMUNICATION_TOPICS
    assert event.fields["engagement"] in COMMUNICATION_ENGAGEMENTS


def test_generate_communication_event_is_reproducible():
    created = datetime(2026, 8, 27, 12, 0)

    first = generate_communication_event(
        behavior=create_behavior(),
        created=created,
        rng=random.Random(42),
    )

    second = generate_communication_event(
        behavior=create_behavior(),
        created=created,
        rng=random.Random(42),
    )

    assert first == second


def test_zero_communication_engagement_produces_no_engagement():
    behavior = create_behavior(
        communication_engagement=0.0,
    )

    for seed in range(100):
        event = generate_communication_event(
            behavior=behavior,
            created=datetime(2026, 8, 27, 12, 0),
            rng=random.Random(seed),
        )

        assert event.fields["engagement"] == "none"
