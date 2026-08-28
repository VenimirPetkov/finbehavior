from datetime import datetime

from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event
from finbehavior.domain.profile import ProfileState
from finbehavior.domain.record import UserRecord
from finbehavior.tokenization.fit import (
    fit_numerical_tokenization,
)
from finbehavior.tokenization.keys import (
    get_event_key_token,
)
from finbehavior.tokenization.numerical import (
    QuantileBucketizer,
)
from finbehavior.tokenization.vocabulary import (
    build_vocabulary,
)


def test_fit_numerical_tokenization_from_records():
    records = [
        UserRecord(
            user_id=1,
            evaluation_point=datetime(2026, 8, 28),
            profile=ProfileState(fields={}),
            events=[
                Event(
                    created=datetime(2026, 8, 27),
                    source=EventSource.TRANSACTION,
                    fields={
                        "amount": 10.0,
                    },
                ),
                Event(
                    created=datetime(2026, 8, 27),
                    source=EventSource.TRANSACTION,
                    fields={
                        "amount": 100.0,
                    },
                ),
                Event(
                    created=datetime(2026, 8, 27),
                    source=EventSource.TRADING,
                    fields={
                        "amount": 500.0,
                    },
                ),
                Event(
                    created=datetime(2026, 8, 27),
                    source=EventSource.TRADING,
                    fields={
                        "amount": 1500.0,
                    },
                ),
            ],
        )
    ]

    vocabulary = build_vocabulary()

    bucketizer = QuantileBucketizer(
        number_of_buckets=4,
    )

    fit_numerical_tokenization(
        records=records,
        bucketizer=bucketizer,
        vocabulary=vocabulary,
    )

    transaction_key = get_event_key_token(
        EventSource.TRANSACTION,
        "amount",
    )

    trading_key = get_event_key_token(
        EventSource.TRADING,
        "amount",
    )

    assert bucketizer.get_boundaries(transaction_key)

    assert bucketizer.get_boundaries(trading_key)

    transaction_bucket = bucketizer.transform(
        transaction_key,
        50.0,
    )

    trading_bucket = bucketizer.transform(
        trading_key,
        1000.0,
    )

    assert vocabulary.get_id(transaction_bucket) >= 0

    assert vocabulary.get_id(trading_bucket) >= 0
