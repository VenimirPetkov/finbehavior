import random

from finbehavior.data.config.profile import (
    BALANCE_QUANTILE_NOISE_VALUES,
    BALANCE_QUANTILE_NOISE_WEIGHTS,
    PLAN_HIGH_SCORE_THRESHOLD,
    PLAN_HIGH_SCORE_WEIGHTS,
    PLAN_LOW_SCORE_WEIGHTS,
    PLAN_MEDIUM_SCORE_THRESHOLD,
    PLAN_MEDIUM_SCORE_WEIGHTS,
    PLAN_SCORE_APP_ACTIVITY_WEIGHT,
    PLAN_SCORE_INCOME_WEIGHT,
    PLAN_SCORE_SPENDING_WEIGHT,
)
from finbehavior.data.reference.field_keys import (
    BALANCE_QUANTILE_FIELD,
    PLAN_FIELD,
    REGION_FIELD,
)
from finbehavior.data.reference.profile import (
    BALANCE_QUANTILE_COUNT,
    BALANCE_QUANTILE_MAX,
    BALANCE_QUANTILE_MIN,
    PLAN_VALUES,
)
from finbehavior.data.reference.regions import REGIONS
from finbehavior.domain.profile import ProfileState

from ..behavior_profile import BehaviorProfile


def choose_plan(
    behavior: BehaviorProfile,
    rng: random.Random,
) -> str:
    score = (
        behavior.income_level * PLAN_SCORE_INCOME_WEIGHT
        + behavior.spending_tendency * PLAN_SCORE_SPENDING_WEIGHT
        + behavior.app_activity * PLAN_SCORE_APP_ACTIVITY_WEIGHT
    )

    if score >= PLAN_HIGH_SCORE_THRESHOLD:
        plan_weights = PLAN_HIGH_SCORE_WEIGHTS

    elif score >= PLAN_MEDIUM_SCORE_THRESHOLD:
        plan_weights = PLAN_MEDIUM_SCORE_WEIGHTS

    else:
        plan_weights = PLAN_LOW_SCORE_WEIGHTS

    return rng.choices(
        population=PLAN_VALUES,
        weights=plan_weights,
        k=1,
    )[0]


def choose_balance_quantile(
    behavior: BehaviorProfile,
    rng: random.Random,
) -> int:
    base_quantile = int(behavior.income_level * BALANCE_QUANTILE_COUNT)

    noise = rng.choices(
        population=BALANCE_QUANTILE_NOISE_VALUES,
        weights=BALANCE_QUANTILE_NOISE_WEIGHTS,
        k=1,
    )[0]

    return max(
        BALANCE_QUANTILE_MIN,
        min(
            BALANCE_QUANTILE_MAX,
            base_quantile + noise,
        ),
    )


def generate_profile_state(
    behavior: BehaviorProfile,
    rng: random.Random | None = None,
) -> ProfileState:
    rng = rng or random.Random()

    return ProfileState(
        fields={
            PLAN_FIELD: choose_plan(
                behavior,
                rng,
            ),
            REGION_FIELD: rng.choice(REGIONS),
            BALANCE_QUANTILE_FIELD: choose_balance_quantile(
                behavior,
                rng,
            ),
        }
    )
