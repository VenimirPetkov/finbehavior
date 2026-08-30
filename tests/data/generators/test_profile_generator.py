import random

from finbehavior.data.behavior_profile import BehaviorProfile
from finbehavior.data.generators.profile import generate_profile_state
from finbehavior.data.reference.field_keys import (
    BALANCE_QUANTILE_FIELD,
    PLAN_FIELD,
    REGION_FIELD,
)
from finbehavior.data.reference.profile import (
    BALANCE_QUANTILE_MAX,
    BALANCE_QUANTILE_MIN,
    PLAN_VALUES,
)
from finbehavior.data.reference.regions import REGIONS


def test_generate_profile_state():
    behavior = BehaviorProfile(
        income_level=0.9,
        spending_tendency=0.5,
        travel_tendency=0.5,
        investing_tendency=0.5,
        app_activity=0.5,
        communication_engagement=0.5,
    )

    profile = generate_profile_state(
        behavior,
        random.Random(42),
    )

    assert profile.fields[PLAN_FIELD] in PLAN_VALUES

    assert profile.fields[REGION_FIELD] in REGIONS

    assert (
        BALANCE_QUANTILE_MIN
        <= profile.fields[BALANCE_QUANTILE_FIELD]
        <= BALANCE_QUANTILE_MAX
    )


def test_generate_profile_state_is_reproducible():
    behavior = BehaviorProfile(
        income_level=0.7,
        spending_tendency=0.6,
        travel_tendency=0.4,
        investing_tendency=0.8,
        app_activity=0.9,
        communication_engagement=0.3,
    )

    first = generate_profile_state(
        behavior,
        random.Random(42),
    )

    second = generate_profile_state(
        behavior,
        random.Random(42),
    )

    assert first == second


def test_balance_quantile_stays_within_bounds():
    low_income_behavior = BehaviorProfile(
        income_level=0.0,
        spending_tendency=0.5,
        travel_tendency=0.5,
        investing_tendency=0.5,
        app_activity=0.5,
        communication_engagement=0.5,
    )

    high_income_behavior = BehaviorProfile(
        income_level=1.0,
        spending_tendency=0.5,
        travel_tendency=0.5,
        investing_tendency=0.5,
        app_activity=0.5,
        communication_engagement=0.5,
    )

    for seed in range(100):
        low_profile = generate_profile_state(
            low_income_behavior,
            random.Random(seed),
        )

        high_profile = generate_profile_state(
            high_income_behavior,
            random.Random(seed),
        )

        assert (
            BALANCE_QUANTILE_MIN
            <= low_profile.fields[BALANCE_QUANTILE_FIELD]
            <= BALANCE_QUANTILE_MAX
        )

        assert (
            BALANCE_QUANTILE_MIN
            <= high_profile.fields[BALANCE_QUANTILE_FIELD]
            <= BALANCE_QUANTILE_MAX
        )
