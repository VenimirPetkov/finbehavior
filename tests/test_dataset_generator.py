from datetime import datetime

import pytest

from finbehavior.data.generators.dataset import (
    generate_dataset,
)


def test_generate_dataset():
    users = generate_dataset(
        number_of_users=10,
        start=datetime(2026, 1, 1),
        evaluation_point=datetime(2026, 7, 1),
        seed=42,
    )

    assert len(users) == 10

    assert [user.record.user_id for user in users] == list(range(10))


def test_generate_dataset_is_reproducible():
    first = generate_dataset(
        number_of_users=5,
        start=datetime(2026, 1, 1),
        evaluation_point=datetime(2026, 7, 1),
        seed=42,
    )

    second = generate_dataset(
        number_of_users=5,
        start=datetime(2026, 1, 1),
        evaluation_point=datetime(2026, 7, 1),
        seed=42,
    )

    assert first == second


def test_different_seeds_generate_different_datasets():
    first = generate_dataset(
        number_of_users=5,
        start=datetime(2026, 1, 1),
        evaluation_point=datetime(2026, 7, 1),
        seed=42,
    )

    second = generate_dataset(
        number_of_users=5,
        start=datetime(2026, 1, 1),
        evaluation_point=datetime(2026, 7, 1),
        seed=43,
    )

    assert first != second


def test_generate_dataset_rejects_zero_users():
    with pytest.raises(
        ValueError,
        match="Number of users must be greater than zero",
    ):
        generate_dataset(
            number_of_users=0,
            start=datetime(2026, 1, 1),
            evaluation_point=datetime(2026, 7, 1),
            seed=42,
        )
