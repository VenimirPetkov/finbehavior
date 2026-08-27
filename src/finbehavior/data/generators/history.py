from finbehavior.data.behavior_profile import BehaviorProfile
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


def calculate_daily_event_rates(
    behavior: BehaviorProfile,
) -> dict[EventSource, float]:
    return {
        EventSource.TRANSACTION: (
            TRANSACTION_BASE_RATE
            + behavior.spending_tendency * TRANSACTION_SPENDING_RATE
        ),
        EventSource.APP: (APP_BASE_RATE + behavior.app_activity * APP_ACTIVITY_RATE),
        EventSource.TRADING: (
            TRADING_BASE_RATE + behavior.investing_tendency * TRADING_ACTIVITY_RATE
        ),
        EventSource.COMMUNICATION: COMMUNICATION_BASE_RATE,
    }
