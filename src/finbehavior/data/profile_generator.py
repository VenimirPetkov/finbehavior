import random

from finbehavior.domain.profile import ProfileState

from .behavior_profile import BehaviorProfile
from .regions import REGIONS


def choose_plan(
    behavior: BehaviorProfile,
    rng: random.Random,
) -> str:
    score = (
        behavior.income_level * 0.5
        + behavior.spending_tendency * 0.3
        + behavior.app_activity * 0.2
    )

    if score >= 0.75:
        return rng.choices(
            population=["basic", "standard", "premium"],
            weights=[0.10, 0.30, 0.60],
            k=1,
        )[0]

    if score >= 0.40:
        return rng.choices(
            population=["basic", "standard", "premium"],
            weights=[0.25, 0.55, 0.20],
            k=1,
        )[0]

    return rng.choices(
        population=["basic", "standard", "premium"],
        weights=[0.70, 0.25, 0.05],
        k=1,
    )[0]


def choose_balance_quantile(
    behavior: BehaviorProfile,
    rng: random.Random,
) -> int:
    base_quantile = int(behavior.income_level * 10)

    noise = rng.choice([-2, -1, 0, 0, 0, 1, 2])

    return max(
        0,
        min(9, base_quantile + noise),
    )


def generate_profile_state(
    behavior: BehaviorProfile,
    rng: random.Random | None = None,
) -> ProfileState:
    rng = rng or random.Random()

    return ProfileState(
        fields={
            "plan": choose_plan(behavior, rng),
            "region": rng.choice(REGIONS),
            "balance_quantile": choose_balance_quantile(
                behavior,
                rng,
            ),
        }
    )
