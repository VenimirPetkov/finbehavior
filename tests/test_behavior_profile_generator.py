import random

from finbehavior.data.behavior_profile_generator import (
    generate_behavior_profile,
)


def test_generate_behavior_profile():
    rng = random.Random(42)

    profile = generate_behavior_profile(rng)

    assert 0.0 <= profile.income_level <= 1.0
    assert 0.0 <= profile.spending_tendency <= 1.0
    assert 0.0 <= profile.travel_tendency <= 1.0
    assert 0.0 <= profile.investing_tendency <= 1.0
    assert 0.0 <= profile.app_activity <= 1.0
    assert 0.0 <= profile.communication_engagement <= 1.0


def test_generate_behavior_profile_is_reproducible():
    first_rng = random.Random(42)
    second_rng = random.Random(42)

    first = generate_behavior_profile(first_rng)
    second = generate_behavior_profile(second_rng)

    assert first == second
