from datetime import datetime
from math import isfinite, log1p, sin, cos, tau

from .config.temporal import (
    DAYS_PER_MONTH_CYCLE,
    DAYS_PER_WEEK,
    HOURS_PER_DAY,
    SOFT_LOG_SCALE_SECONDS,
)


def soft_log_elapsed_seconds(
    elapsed_seconds: float,
) -> float:
    if not isfinite(elapsed_seconds):
        raise ValueError(
            "Elapsed seconds must be finite"
        )

    if elapsed_seconds < 0:
        raise ValueError(
            "Elapsed seconds cannot be negative"
        )

    return (
        SOFT_LOG_SCALE_SECONDS
        * log1p(
            elapsed_seconds
            / SOFT_LOG_SCALE_SECONDS
        )
    )


def seconds_to_latest_event(
    event_time: datetime,
    latest_event_time: datetime,
) -> float:
    if event_time > latest_event_time:
        raise ValueError(
            "Event time cannot be after latest event time"
        )

    return (
        latest_event_time - event_time
    ).total_seconds()


def encode_cycle(
    value: float,
    period: float,
) -> tuple[float, float]:
    angle = tau * value / period

    return (
        sin(angle),
        cos(angle),
    )


def get_calendar_features(
    timestamp: datetime,
) -> tuple[float, ...]:
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    day_of_month = timestamp.day - 1

    hour_sin, hour_cos = encode_cycle(
        hour,
        HOURS_PER_DAY,
    )

    weekday_sin, weekday_cos = encode_cycle(
        day_of_week,
        DAYS_PER_WEEK,
    )

    monthday_sin, monthday_cos = encode_cycle(
        day_of_month,
        DAYS_PER_MONTH_CYCLE,
    )

    return (
        hour_sin,
        hour_cos,
        weekday_sin,
        weekday_cos,
        monthday_sin,
        monthday_cos,
    )