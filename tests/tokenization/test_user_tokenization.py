from datetime import datetime

import pytest

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


def test_events_are_sorted_chronologically_without_mutating_record():
    later_event = Event(
        created=datetime(2026, 8, 27, 18, 0),
        source=EventSource.APP,
        fields={"screen": "cards"},
    )
    earlier_event = Event(
        created=datetime(2026, 8, 27, 12, 0),
        source=EventSource.APP,
        fields={"screen": "home"},
    )
    record = UserRecord(
        user_id=42,
        evaluation_point=datetime(2026, 8, 27, 20, 0),
        profile=ProfileState(fields={}),
        events=[later_event, earlier_event],
    )
    vocabulary = build_vocabulary()

    tokenized = tokenize_user_record(
        record=record,
        vocabulary=vocabulary,
        numerical_bucketizer=QuantileBucketizer(number_of_buckets=4),
    )

    first_value = tokenized.events[0].fields[0].value_id
    second_value = tokenized.events[1].fields[0].value_id

    assert vocabulary.get_token(first_value) == "home"
    assert vocabulary.get_token(second_value) == "cards"
    assert record.events == [later_event, earlier_event]


def test_max_events_keeps_only_most_recent_events():
    events = [
        Event(
            created=datetime(2026, 8, 27, hour, 0),
            source=EventSource.APP,
            fields={"screen": screen},
        )
        for hour, screen in ((10, "home"), (12, "cards"), (14, "profile"))
    ]
    record = UserRecord(
        user_id=42,
        evaluation_point=datetime(2026, 8, 27, 20, 0),
        profile=ProfileState(fields={}),
        events=events,
    )
    vocabulary = build_vocabulary()

    tokenized = tokenize_user_record(
        record=record,
        vocabulary=vocabulary,
        numerical_bucketizer=QuantileBucketizer(number_of_buckets=4),
        max_events=2,
    )

    screens = tuple(
        vocabulary.get_token(event.fields[0].value_id) for event in tokenized.events
    )
    assert screens == ("cards", "profile")


def test_max_events_must_be_positive():
    record = UserRecord(
        user_id=42,
        evaluation_point=datetime(2026, 8, 27, 20, 0),
        profile=ProfileState(fields={}),
        events=[],
    )

    with pytest.raises(ValueError, match="max_events must be at least 1"):
        tokenize_user_record(
            record=record,
            vocabulary=build_vocabulary(),
            numerical_bucketizer=QuantileBucketizer(number_of_buckets=4),
            max_events=0,
        )
