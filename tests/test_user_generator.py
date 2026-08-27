import random
from datetime import datetime

from finbehavior.data.generators.user import (
    generate_synthetic_user,
)


def test_generate_synthetic_user():
    start = datetime(2026, 1, 1)
    evaluation_point = datetime(2026, 7, 1)

    user = generate_synthetic_user(
        user_id=42,
        start=start,
        evaluation_point=evaluation_point,
        rng=random.Random(42),
    )

    assert user.record.user_id == 42

    assert user.record.evaluation_point == evaluation_point

    assert len(user.record.events) > 0

    assert all(event.created < evaluation_point for event in user.record.events)


def test_generate_synthetic_user_is_reproducible():
    start = datetime(2026, 1, 1)
    evaluation_point = datetime(2026, 7, 1)

    first = generate_synthetic_user(
        user_id=42,
        start=start,
        evaluation_point=evaluation_point,
        rng=random.Random(42),
    )

    second = generate_synthetic_user(
        user_id=42,
        start=start,
        evaluation_point=evaluation_point,
        rng=random.Random(42),
    )

    assert first == second
