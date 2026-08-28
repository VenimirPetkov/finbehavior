from finbehavior.tokenization.numerical import (
    QuantileBucketizer,
)
from finbehavior.tokenization.vocabulary import (
    Vocabulary,
    build_vocabulary,
)


def test_vocabulary_round_trip():
    vocabulary = build_vocabulary()

    original_tokens = vocabulary.get_tokens()

    restored = Vocabulary.from_tokens(original_tokens)

    assert restored.get_tokens() == (original_tokens)

    assert len(restored) == len(vocabulary)


def test_quantile_bucketizer_round_trip():
    bucketizer = QuantileBucketizer(
        number_of_buckets=4,
    )

    bucketizer.fit(
        "transaction.amount",
        (
            10,
            20,
            30,
            40,
            50,
        ),
    )

    original_boundaries = bucketizer.get_all_boundaries()

    restored = QuantileBucketizer.from_boundaries(
        number_of_buckets=4,
        boundaries=original_boundaries,
    )

    assert restored.get_all_boundaries() == original_boundaries

    assert restored.transform(
        "transaction.amount",
        25,
    ) == bucketizer.transform(
        "transaction.amount",
        25,
    )
