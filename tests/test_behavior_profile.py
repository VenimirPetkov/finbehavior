import pytest
from finbehavior.data.behavior_profile import BehaviorProfile


def test_create_behavior_profile():
    profile = BehaviorProfile(
        income_level=0.8,
        spending_tendency=0.6,
        travel_tendency=0.9,
        investing_tendency=0.7,
        app_activity=0.8,
        communication_engagement=0.3,
    )

    assert profile.income_level == 0.8
    assert profile.spending_tendency == 0.6
    assert profile.travel_tendency == 0.9
    assert profile.investing_tendency == 0.7
    assert profile.app_activity == 0.8
    assert profile.communication_engagement == 0.3


def test_behavior_profile_rejects_values_above_one():
    with pytest.raises(ValueError):
        BehaviorProfile(
            income_level=1.5,
            spending_tendency=0.6,
            travel_tendency=0.9,
            investing_tendency=0.7,
            app_activity=0.8,
            communication_engagement=0.3,
        )


def test_behavior_profile_rejects_negative_values():
    with pytest.raises(ValueError):
        BehaviorProfile(
            income_level=0.8,
            spending_tendency=-0.2,
            travel_tendency=0.9,
            investing_tendency=0.7,
            app_activity=0.8,
            communication_engagement=0.3,
        )
