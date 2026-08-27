import random
from datetime import datetime

from finbehavior.data.generators.user import (
    generate_synthetic_user,
)
from finbehavior.data.synthetic_user import SyntheticUser


def generate_dataset(
    number_of_users: int,
    start: datetime,
    evaluation_point: datetime,
    seed: int | None = None,
) -> list[SyntheticUser]:
    if number_of_users <= 0:
        raise ValueError("Number of users must be greater than zero")

    rng = random.Random(seed)

    users: list[SyntheticUser] = []

    for user_id in range(number_of_users):
        user_seed = rng.randrange(0, 2**32)

        user = generate_synthetic_user(
            user_id=user_id,
            start=start,
            evaluation_point=evaluation_point,
            rng=random.Random(user_seed),
        )

        users.append(user)

    return users
