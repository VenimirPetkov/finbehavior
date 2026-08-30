from datetime import datetime

from finbehavior.data.reference.field_keys import (
    AMOUNT_FIELD,
)
from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event
from finbehavior.domain.profile import ProfileState
from finbehavior.domain.record import UserRecord
from finbehavior.tokenization.keys import (
    get_event_key_token,
)
from finbehavior.tokenization.numerical import (
    QuantileBucketizer,
)
from finbehavior.tokenization.user import (
    tokenize_user_record,
)
from finbehavior.tokenization.vocabulary import (
    build_vocabulary,
)

TRANSACTION_AMOUNTS = (
    5,
    8,
    12,
    18,
    24,
    31,
    45,
    70,
    90,
    120,
    160,
    210,
    280,
    350,
    420,
    550,
    700,
    950,
    1400,
    2500,
)


def test_tokenize_user_record():
    profile = ProfileState(
        fields={
            "plan": "premium",
            "region": "ES",
            "balance_quantile": 7,
        }
    )

    events = [
        Event(
            created=datetime(2026, 8, 27, 12, 0),
            source=EventSource.TRANSACTION,
            fields={
                "type": "card_payment",
                "direction": "out",
                "amount": 42.50,
                "currency": "EUR",
                "merchant_category": "restaurant",
                "merchant_region": "ES",
            },
        ),
        Event(
            created=datetime(2026, 8, 27, 18, 0),
            source=EventSource.TRANSACTION,
            fields={
                "type": "card_payment",
                "direction": "out",
                "amount": 120.00,
                "currency": "EUR",
                "merchant_category": "restaurant",
                "merchant_region": "ES",
            },
        ),
    ]

    record = UserRecord(
        user_id=42,
        evaluation_point=datetime(2026, 8, 27, 20, 0),
        profile=profile,
        events=events,
    )

    vocabulary = build_vocabulary()

    amount_key = get_event_key_token(
        EventSource.TRANSACTION,
        AMOUNT_FIELD,
    )

    bucketizer = QuantileBucketizer(
        number_of_buckets=4,
    )

    bucketizer.fit(
        amount_key,
        TRANSACTION_AMOUNTS,
    )

    vocabulary.add_many(bucketizer.get_bucket_tokens(amount_key))

    tokenized = tokenize_user_record(
        record=record,
        vocabulary=vocabulary,
        numerical_bucketizer=bucketizer,
    )

    assert tokenized.user_id == 42

    assert len(tokenized.profile.fields) == 3

    assert len(tokenized.events) == 2

    assert tokenized.events[0].elapsed_time_feature > 0

    assert tokenized.events[1].elapsed_time_feature == 0


def test_tokenize_user_without_events():
    record = UserRecord(
        user_id=42,
        evaluation_point=datetime(2026, 8, 27, 20, 0),
        profile=ProfileState(
            fields={
                "plan": "premium",
                "region": "ES",
                "balance_quantile": 7,
            }
        ),
        events=[],
    )

    tokenized = tokenize_user_record(
        record=record,
        vocabulary=build_vocabulary(),
        numerical_bucketizer=QuantileBucketizer(number_of_buckets=4),
    )

    assert tokenized.user_id == 42
    assert tokenized.events == ()
