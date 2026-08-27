from datetime import datetime
from math import log

import pytest

from finbehavior.tokenization.temporal import (
    encode_cycle,
    get_calendar_features,
    seconds_to_latest_event,
    soft_log_elapsed_seconds,
)


def test_seconds_to_latest_event():
    event_time = datetime(2026, 8, 27, 15, 0)

    latest_event_time = datetime(2026, 8, 27, 18, 0)

    seconds = seconds_to_latest_event(
        event_time,
        latest_event_time,
    )

    assert seconds == 3 * 60 * 60


def test_latest_event_has_zero_elapsed_time():
    timestamp = datetime(2026, 8, 27, 18, 0)

    assert (
        seconds_to_latest_event(
            timestamp,
            timestamp,
        )
        == 0
    )


def test_soft_log_zero_is_zero():
    assert soft_log_elapsed_seconds(0) == 0


def test_soft_log_uses_pragma_transformation():
    result = soft_log_elapsed_seconds(8)

    assert result == pytest.approx(8 * log(2))


def test_encode_cycle_wraps_around():
    start_sin, start_cos = encode_cycle(
        0,
        24,
    )

    end_sin, end_cos = encode_cycle(
        24,
        24,
    )

    assert end_sin == pytest.approx(start_sin)

    assert end_cos == pytest.approx(start_cos)


def test_calendar_features_have_six_values():
    features = get_calendar_features(datetime(2026, 8, 27, 14, 30))

    assert len(features) == 6


def test_rejects_event_after_latest_event():
    with pytest.raises(
        ValueError,
        match="Event time cannot be after latest event time",
    ):
        seconds_to_latest_event(
            datetime(2026, 8, 28),
            datetime(2026, 8, 27),
        )
