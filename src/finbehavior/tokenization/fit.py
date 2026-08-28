from collections import defaultdict
from collections.abc import Sequence

from finbehavior.data.reference.field_keys import (
    NUMERICAL_FIELD_KEYS_BY_SOURCE,
)
from finbehavior.domain.record import UserRecord

from .keys import get_event_key_token
from .numerical import QuantileBucketizer
from .vocabulary import Vocabulary


def fit_numerical_tokenization(
    records: Sequence[UserRecord],
    bucketizer: QuantileBucketizer,
    vocabulary: Vocabulary,
) -> None:
    values_by_key: dict[str, list[int | float]] = defaultdict(list)

    for record in records:
        for event in record.events:
            numerical_fields = NUMERICAL_FIELD_KEYS_BY_SOURCE.get(
                event.source,
                (),
            )

            for field_name in numerical_fields:
                if field_name not in event.fields:
                    continue

                value = event.fields[field_name]

                if isinstance(value, bool) or not isinstance(
                    value,
                    (int, float),
                ):
                    raise TypeError(
                        f"Numerical field '{field_name}' " "must contain a number"
                    )

                key_token = get_event_key_token(
                    event.source,
                    field_name,
                )

                values_by_key[key_token].append(value)

    for key_token, values in values_by_key.items():
        bucketizer.fit(
            key_token,
            values,
        )

        vocabulary.add_many(bucketizer.get_bucket_tokens(key_token))
