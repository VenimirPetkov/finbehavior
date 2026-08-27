from finbehavior.data.behavior_profile import BehaviorProfile
from finbehavior.data.generators.history import (
    calculate_daily_event_rates,
)
from finbehavior.domain.enums import EventSource


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

    assert rates[EventSource.TRANSACTION] == 2.1
    assert rates[EventSource.APP] == 5.0
    assert rates[EventSource.TRADING] == 0.3
    assert rates[EventSource.COMMUNICATION] == 0.08


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

    assert high_rates[EventSource.TRANSACTION] > low_rates[EventSource.TRANSACTION]

    assert high_rates[EventSource.APP] > low_rates[EventSource.APP]

    assert high_rates[EventSource.TRADING] > low_rates[EventSource.TRADING]
