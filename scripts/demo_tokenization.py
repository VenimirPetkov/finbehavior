from datetime import datetime

from finbehavior.data.reference.field_keys import (
    AMOUNT_FIELD,
    CURRENCY_FIELD,
    DIRECTION_FIELD,
    MERCHANT_CATEGORY_FIELD,
    MERCHANT_REGION_FIELD,
    TYPE_FIELD,
)
from finbehavior.data.reference.transaction import (
    TRANSACTION_DIRECTION_OUT,
    TRANSACTION_TYPE_CARD_PAYMENT,
)
from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event
from finbehavior.tokenization.event import tokenize_event
from finbehavior.tokenization.keys import get_event_key_token
from finbehavior.tokenization.numerical import QuantileBucketizer
from finbehavior.tokenization.vocabulary import build_vocabulary


DEMO_TRANSACTION_AMOUNTS = (
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

DEMO_EVENT_TIME = datetime(
    2026,
    8,
    27,
    14,
    30,
)

DEMO_LATEST_EVENT_TIME = datetime(
    2026,
    8,
    27,
    18,
    0,
)


def main() -> None:
    event = Event(
        created=DEMO_EVENT_TIME,
        source=EventSource.TRANSACTION,
        fields={
            TYPE_FIELD: TRANSACTION_TYPE_CARD_PAYMENT,
            DIRECTION_FIELD: TRANSACTION_DIRECTION_OUT,
            AMOUNT_FIELD: 42.50,
            CURRENCY_FIELD: "EUR",
            MERCHANT_CATEGORY_FIELD: "restaurant",
            MERCHANT_REGION_FIELD: "ES",
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
        DEMO_TRANSACTION_AMOUNTS,
    )

    vocabulary.add_many(
        bucketizer.get_bucket_tokens(
            amount_key
        )
    )

    tokenized = tokenize_event(
        event=event,
        latest_event_time=DEMO_LATEST_EVENT_TIME,
        vocabulary=vocabulary,
        numerical_bucketizer=bucketizer,
    )

    print_raw_event(event)

    print_numerical_step(
        event=event,
        amount_key=amount_key,
        bucketizer=bucketizer,
    )

    print_tokenized_event(
        tokenized=tokenized,
        vocabulary=vocabulary,
    )


def print_raw_event(
    event: Event,
) -> None:
    print()
    print("RAW EVENT")
    print("=" * 50)

    print(f"created: {event.created}")
    print(f"source: {event.source.value}")

    for key, value in event.fields.items():
        print(f"{key}: {value}")


def print_numerical_step(
    event: Event,
    amount_key: str,
    bucketizer: QuantileBucketizer,
) -> None:
    amount = event.fields[AMOUNT_FIELD]

    bucket_token = bucketizer.transform(
        amount_key,
        amount,
    )

    print()
    print("NUMERICAL")
    print("=" * 50)

    print(f"{amount_key} boundaries:")

    for boundary in bucketizer.get_boundaries(
        amount_key
    ):
        print(f"  {boundary:.2f}")

    print()
    print(
        f"{amount} -> {bucket_token}"
    )


def print_tokenized_event(
    tokenized,
    vocabulary,
) -> None:
    print()
    print("TOKENIZED EVENT")
    print("=" * 50)

    event_token = vocabulary.get_token(
        tokenized.event_token_id
    )

    print(
        f"{event_token} -> "
        f"{tokenized.event_token_id}"
    )

    for field in tokenized.fields:
        key_token = vocabulary.get_token(
            field.key_id
        )

        value_token = vocabulary.get_token(
            field.value_id
        )

        print()
        print(
            f"{key_token:<35} -> "
            f"{field.key_id}"
        )

        print(
            f"{value_token:<35} -> "
            f"{field.value_id}"
        )

    print()
    print("TEMPORAL")
    print("=" * 50)

    print(
        "elapsed feature: "
        f"{tokenized.elapsed_time_feature:.6f}"
    )

    print("calendar features:")

    for index, value in enumerate(
        tokenized.calendar_features
    ):
        print(
            f"  feature_{index}: "
            f"{value:.6f}"
        )


if __name__ == "__main__":
    main()