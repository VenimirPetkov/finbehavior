import random
from dataclasses import dataclass

from finbehavior.domain.record import UserRecord

from .config.split import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_FRACTION,
)


@dataclass(frozen=True)
class UserRecordSplit:
    train_records: tuple[UserRecord, ...]
    validation_records: tuple[UserRecord, ...]


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
