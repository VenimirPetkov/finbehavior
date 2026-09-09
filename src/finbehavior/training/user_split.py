import random
from dataclasses import dataclass

from finbehavior.domain.record import UserRecord

from .config.split import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_VALIDATION_FRACTION,
)


@dataclass(frozen=True)
class UserRecordSplit:
    train_records: tuple[UserRecord, ...]
    validation_records: tuple[UserRecord, ...]


@dataclass(frozen=True)
class UserRecordThreeWaySplit:
    train_records: tuple[UserRecord, ...]
    validation_records: tuple[UserRecord, ...]
    test_records: tuple[UserRecord, ...]


def split_user_records(
    records: tuple[UserRecord, ...],
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> UserRecordSplit:
    if len(records) < 2:
        raise ValueError("At least two user records are required")

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("Train fraction must be between zero and one")

    shuffled_records = list(records)

    rng = random.Random(seed)
    rng.shuffle(shuffled_records)

    train_count = int(len(shuffled_records) * train_fraction)

    train_count = max(
        1,
        min(
            train_count,
            len(shuffled_records) - 1,
        ),
    )

    return UserRecordSplit(
        train_records=tuple(shuffled_records[:train_count]),
        validation_records=tuple(shuffled_records[train_count:]),
    )


def split_user_records_three_way(
    records: tuple[UserRecord, ...],
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> UserRecordThreeWaySplit:
    """Deterministically split users into disjoint train, validation, and test sets.

    The legacy :func:`split_user_records` remains a two-way split so existing callers
    keep their previous behavior. New experiments should use this function and keep
    the test records untouched until model selection is complete.
    """
    if len(records) < 3:
        raise ValueError("At least three user records are required")

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("Train fraction must be between zero and one")

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("Validation fraction must be between zero and one")

    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train and validation fractions must sum to less than one")

    shuffled_records = list(records)

    rng = random.Random(seed)
    rng.shuffle(shuffled_records)

    train_count = max(1, int(len(shuffled_records) * train_fraction))
    validation_count = max(1, int(len(shuffled_records) * validation_fraction))

    while train_count + validation_count > len(shuffled_records) - 1:
        if train_count >= validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            raise ValueError("Split fractions do not leave room for a test record")

    validation_end = train_count + validation_count

    return UserRecordThreeWaySplit(
        train_records=tuple(shuffled_records[:train_count]),
        validation_records=tuple(shuffled_records[train_count:validation_end]),
        test_records=tuple(shuffled_records[validation_end:]),
    )
