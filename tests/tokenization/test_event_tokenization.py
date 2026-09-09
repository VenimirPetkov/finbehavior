from datetime import datetime

import pytest

from finbehavior.data.reference.field_keys import (
    AMOUNT_FIELD,
)
from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event
from finbehavior.tokenization.event import (
    tokenize_event,
)
from finbehavior.tokenization.keys import (
    get_event_key_token,
)
from finbehavior.tokenization.numerical import (
    QuantileBucketizer,
)
from finbehavior.tokenization.special_tokens import (
    EVT_TOKEN,
    UNK_TOKEN,
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


def test_tokenize_transaction_event():
    event = Event(
        created=datetime(
            2026,
            8,
            27,
            14,
            30,
        ),
        source=EventSource.TRANSACTION,
        fields={
            "type": "card_payment",
            "direction": "out",
            "amount": 42.50,
            "currency": "EUR",
            "merchant_category": "restaurant",
            "merchant_region": "ES",
        },
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

    tokenized = tokenize_event(
        event=event,
        latest_event_time=datetime(
            2026,
            8,
            27,
            18,
            0,
        ),
        vocabulary=vocabulary,
        numerical_bucketizer=bucketizer,
    )

    assert tokenized.event_token_id == (vocabulary.get_id(EVT_TOKEN))

    assert len(tokenized.fields) == 6

    assert len(tokenized.calendar_features) == 6

    assert tokenized.elapsed_time_feature > 0


def test_cross_field_categorical_value_is_encoded_as_unknown():
    event = Event(
        created=datetime(2026, 8, 27, 14, 30),
        source=EventSource.APP,
        fields={
            "screen": "opened",
            "action": "opened",
        },
    )
    vocabulary = build_vocabulary()

    tokenized = tokenize_event(
        event=event,
        latest_event_time=event.created,
        vocabulary=vocabulary,
        numerical_bucketizer=QuantileBucketizer(number_of_buckets=4),
    )

    values_by_key = {
        vocabulary.get_token(field.key_id): field.value_id for field in tokenized.fields
    }

    assert values_by_key["app.screen"] == vocabulary.get_id(UNK_TOKEN)
    assert values_by_key["app.action"] == vocabulary.get_id("opened")


def test_numerical_field_rejects_globally_known_string_value():
    event = Event(
        created=datetime(2026, 8, 27, 14, 30),
        source=EventSource.TRANSACTION,
        fields={"amount": "EUR"},
    )

    with pytest.raises(TypeError, match="must contain a number"):
        tokenize_event(
            event=event,
            latest_event_time=event.created,
            vocabulary=build_vocabulary(),
            numerical_bucketizer=QuantileBucketizer(number_of_buckets=4),
        )
