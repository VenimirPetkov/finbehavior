import random
from datetime import datetime

import pytest

from finbehavior.data.generators.timeline import (
    generate_event_timestamps,
)


def test_generate_event_timestamps():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 2, 1)

    timestamps = generate_event_timestamps(
        start=start,
        end=end,
        daily_rate=2.0,
        rng=random.Random(42),
    )

    assert len(timestamps) > 0

    assert all(start < timestamp < end for timestamp in timestamps)

    assert timestamps == sorted(timestamps)


def test_generate_event_timestamps_is_reproducible():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 2, 1)

    first = generate_event_timestamps(
        start=start,
        end=end,
        daily_rate=2.0,
        rng=random.Random(42),
    )

    second = generate_event_timestamps(
        start=start,
        end=end,
        daily_rate=2.0,
        rng=random.Random(42),
    )

    assert first == second


def test_zero_daily_rate_generates_no_timestamps():
    timestamps = generate_event_timestamps(
        start=datetime(2026, 1, 1),
        end=datetime(2026, 2, 1),
        daily_rate=0.0,
        rng=random.Random(42),
    )

    assert timestamps == []


def test_rejects_invalid_date_range():
    with pytest.raises(ValueError):
        generate_event_timestamps(
            start=datetime(2026, 2, 1),
            end=datetime(2026, 1, 1),
            daily_rate=1.0,
            rng=random.Random(42),
        )

def test_rejects_negative_daily_rate():
    with pytest.raises(ValueError):
        generate_event_timestamps(
            start=datetime(2026, 1, 1),
            end=datetime(2026, 2, 1),
            daily_rate=-1.0,
            rng=random.Random(42),
        )