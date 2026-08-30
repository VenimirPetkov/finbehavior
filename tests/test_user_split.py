from datetime import datetime

import pytest

from finbehavior.domain.profile import ProfileState
from finbehavior.domain.record import UserRecord
from finbehavior.training.user_split import (
    split_user_records,
)


def build_record(
    user_id: int,
) -> UserRecord:
    return UserRecord(
        user_id=user_id,
        evaluation_point=datetime(
            2026,
            1,
            15,
        ),
        profile=ProfileState(
            fields={},
        ),
        events=[],
    )


def test_split_user_records_separates_train_and_validation_users():
    records = tuple(build_record(user_id) for user_id in range(10))

    split = split_user_records(
        records=records,
        train_fraction=0.8,
        seed=42,
    )

    assert len(split.train_records) == 8
    assert len(split.validation_records) == 2

    train_user_ids = {record.user_id for record in split.train_records}

    validation_user_ids = {record.user_id for record in split.validation_records}

    assert train_user_ids.isdisjoint(validation_user_ids)

    assert (train_user_ids | validation_user_ids) == set(range(10))


def test_split_user_records_is_deterministic():
    records = tuple(build_record(user_id) for user_id in range(10))

    first_split = split_user_records(
        records=records,
        seed=42,
    )

    second_split = split_user_records(
        records=records,
        seed=42,
    )

    first_train_ids = tuple(record.user_id for record in first_split.train_records)

    second_train_ids = tuple(record.user_id for record in second_split.train_records)

    assert first_train_ids == second_train_ids


def test_split_user_records_rejects_invalid_train_fraction():
    records = (
        build_record(1),
        build_record(2),
    )

    with pytest.raises(
        ValueError,
        match="Train fraction must be between zero and one",
    ):
        split_user_records(
            records=records,
            train_fraction=1.0,
        )


def test_split_user_records_requires_at_least_two_users():
    with pytest.raises(
        ValueError,
        match="At least two user records are required",
    ):
        split_user_records(
            records=(build_record(1),),
        )
