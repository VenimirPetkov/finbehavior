import random

from .behavior_profile import BehaviorProfile


def generate_behavior_profile(
    rng: random.Random | None = None,
) -> BehaviorProfile:
    rng = rng or random.Random()

    return BehaviorProfile(
        income_level=rng.random(),
        spending_tendency=rng.random(),
        travel_tendency=rng.random(),
        investing_tendency=rng.random(),
        app_activity=rng.random(),
        communication_engagement=rng.random(),
    )
