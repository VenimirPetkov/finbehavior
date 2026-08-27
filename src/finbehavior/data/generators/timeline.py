import random
from datetime import datetime, timedelta


def generate_event_timestamps(
    start: datetime,
    end: datetime,
    daily_rate: float,
    rng: random.Random | None = None,
) -> list[datetime]:
    if end <= start:
        raise ValueError("End must be after start")

    if daily_rate < 0:
        raise ValueError("Daily rate cannot be negative")

    if daily_rate == 0:
        return []

    rng = rng or random.Random()

    timestamps: list[datetime] = []
    current = start

    while True:
        wait_days = rng.expovariate(daily_rate)

        current = current + timedelta(
            days=wait_days,
        )

        if current >= end:
            break

        timestamps.append(current)

    return timestamps
