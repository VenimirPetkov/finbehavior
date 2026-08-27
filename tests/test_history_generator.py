import random
from datetime import datetime

import pytest

from finbehavior.data.behavior_profile import BehaviorProfile
from finbehavior.data.generators.history import (
    calculate_daily_event_rates,
    generate_event_history,
)
from finbehavior.domain.enums import EventSource
from finbehavior.domain.profile import ProfileState


def create_behavior() -> BehaviorProfile:
    return BehaviorProfile(
        income_level=0.7,
        spending_tendency=0.8,
        travel_tendency=0.6,
        investing_tendency=0.7,
        app_activity=0.8,
        communication_engagement=0.5,
    )


def create_profile() -> ProfileState:
    return ProfileState(
        fields={
            "plan": "premium",
            "region": "ES",
            "balance_quantile": 7,
        }
    )


def test_calculate_daily_event_rates():
    behavior = BehaviorProfile(
        income_level=0.7,
        spending_tendency=0.8,
        travel_tendency=0.5,
        investing_tendency=0.6,
        app_activity=0.9,
        communication_engagement=0.7,
    )

    rates = calculate_daily_event_rates(behavior)

    assert rates[EventSource.TRANSACTION] == pytest.approx(2.1)
    assert rates[EventSource.APP] == pytest.approx(5.0)
    assert rates[EventSource.TRADING] == pytest.approx(0.3)
    assert rates[EventSource.COMMUNICATION] == pytest.approx(0.08)


def test_higher_activity_increases_relevant_event_rates():
    low_activity = BehaviorProfile(
        income_level=0.5,
        spending_tendency=0.1,
        travel_tendency=0.5,
        investing_tendency=0.1,
        app_activity=0.1,
        communication_engagement=0.5,
    )

    high_activity = BehaviorProfile(
        income_level=0.5,
        spending_tendency=0.9,
        travel_tendency=0.5,
        investing_tendency=0.9,
        app_activity=0.9,
        communication_engagement=0.5,
    )

    low_rates = calculate_daily_event_rates(low_activity)
    high_rates = calculate_daily_event_rates(high_activity)

    assert (
        high_rates[EventSource.TRANSACTION]
        > low_rates[EventSource.TRANSACTION]
    )

    assert (
        high_rates[EventSource.APP]
        > low_rates[EventSource.APP]
    )

    assert (
        high_rates[EventSource.TRADING]
        > low_rates[EventSource.TRADING]
    )


def test_generate_event_history():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 2, 1)

    events = generate_event_history(
        behavior=create_behavior(),
        profile=create_profile(),
        start=start,
        end=end,
        rng=random.Random(42),
    )

    assert len(events) > 0

    assert all(
        start < event.created < end
        for event in events
    )

    assert events == sorted(
        events,
        key=lambda event: event.created,
    )

    assert all(
        isinstance(event.source, EventSource)
        for event in events
    )


def test_generate_event_history_is_reproducible():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 2, 1)

    first = generate_event_history(
        behavior=create_behavior(),
        profile=create_profile(),
        start=start,
        end=end,
        rng=random.Random(42),
    )

    second = generate_event_history(
        behavior=create_behavior(),
        profile=create_profile(),
        start=start,
        end=end,
        rng=random.Random(42),
    )

    assert first == second


def test_generate_event_history_requires_region():
    profile = ProfileState(
        fields={
            "plan": "standard",
            "balance_quantile": 5,
        }
    )

    with pytest.raises(
        ValueError,
        match="Profile must contain a valid region",
    ):
        generate_event_history(
            behavior=create_behavior(),
            profile=profile,
            start=datetime(2026, 1, 1),
            end=datetime(2026, 2, 1),
            rng=random.Random(42),
        )