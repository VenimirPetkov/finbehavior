import pytest

from finbehavior.data.reference.field_keys import (
    AMOUNT_FIELD,
)
from finbehavior.domain.enums import EventSource
from finbehavior.tokenization.keys import (
    get_event_key_token,
)
from finbehavior.tokenization.numerical import (
    QuantileBucketizer,
)
from finbehavior.tokenization.vocabulary import build_vocabulary

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


def test_fit_quantile_boundaries():
    key_token = get_event_key_token(
        EventSource.TRANSACTION,
        AMOUNT_FIELD,
    )

    bucketizer = QuantileBucketizer(
        number_of_buckets=4,
    )

    bucketizer.fit(
        key_token,
        TRANSACTION_AMOUNTS,
    )

    assert bucketizer.get_boundaries(key_token) == pytest.approx(
        (
            29.25,
            140.0,
            452.5,
        )
    )


def test_transform_value_to_bucket_token():
    key_token = get_event_key_token(
        EventSource.TRANSACTION,
        AMOUNT_FIELD,
    )

    bucketizer = QuantileBucketizer(
        number_of_buckets=4,
    )

    bucketizer.fit(
        key_token,
        TRANSACTION_AMOUNTS,
    )

    token = bucketizer.transform(
        key_token,
        428.73,
    )

    assert token == "transaction.amount.bucket_2"


def test_same_amount_can_have_different_bucket_by_source():
    transaction_key = get_event_key_token(
        EventSource.TRANSACTION,
        AMOUNT_FIELD,
    )

    trading_key = get_event_key_token(
        EventSource.TRADING,
        AMOUNT_FIELD,
    )

    trading_amounts = (
        100,
        200,
        300,
        400,
        500,
        1000,
        2000,
        3000,
    )

    bucketizer = QuantileBucketizer(
        number_of_buckets=4,
    )

    bucketizer.fit(
        transaction_key,
        TRANSACTION_AMOUNTS,
    )

    bucketizer.fit(
        trading_key,
        trading_amounts,
    )

    transaction_token = bucketizer.transform(
        transaction_key,
        428.73,
    )

    trading_token = bucketizer.transform(
        trading_key,
        428.73,
    )

    assert transaction_token == "transaction.amount.bucket_2"
    assert trading_token == "trading.amount.bucket_1"


def test_transform_requires_fitted_key():
    bucketizer = QuantileBucketizer(
        number_of_buckets=4,
    )

    with pytest.raises(
        ValueError,
        match="No quantile boundaries fitted",
    ):
        bucketizer.transform(
            "transaction.amount",
            100,
        )


def test_rejects_invalid_bucket_count():
    with pytest.raises(
        ValueError,
        match="Number of buckets must be at least 2",
    ):
        QuantileBucketizer(
            number_of_buckets=1,
        )
        
def test_numerical_bucket_token_can_be_registered_in_vocabulary():
    key_token = get_event_key_token(
        EventSource.TRANSACTION,
        AMOUNT_FIELD,
    )

    bucketizer = QuantileBucketizer(
        number_of_buckets=4,
    )

    bucketizer.fit(
        key_token,
        TRANSACTION_AMOUNTS,
    )

    vocab = build_vocabulary()

    vocab.add_many(
        bucketizer.get_bucket_tokens(
            key_token
        )
    )

    bucket_token = bucketizer.transform(
        key_token,
        428.73,
    )

    token_id = vocab.get_id(
        bucket_token
    )

    assert bucket_token == "transaction.amount.bucket_2"

    assert vocab.get_token(
        token_id
    ) == bucket_token