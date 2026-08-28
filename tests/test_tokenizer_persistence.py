from pathlib import Path

from finbehavior.tokenization.numerical import (
    QuantileBucketizer,
)
from finbehavior.tokenization.persistence import (
    load_tokenizer,
    save_tokenizer,
)
from finbehavior.tokenization.vocabulary import (
    build_vocabulary,
)


def test_save_and_load_tokenizer(
    tmp_path: Path,
):
    vocabulary = build_vocabulary()

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

    vocabulary.add_many(bucketizer.get_bucket_tokens("transaction.amount"))

    path = tmp_path / "tokenizer.json"

    save_tokenizer(
        path=path,
        vocabulary=vocabulary,
        bucketizer=bucketizer,
    )

    (
        restored_vocabulary,
        restored_bucketizer,
    ) = load_tokenizer(path)

    assert restored_vocabulary.get_tokens() == vocabulary.get_tokens()

    assert restored_bucketizer.get_all_boundaries() == bucketizer.get_all_boundaries()

    original_bucket = bucketizer.transform(
        "transaction.amount",
        25,
    )

    restored_bucket = restored_bucketizer.transform(
        "transaction.amount",
        25,
    )

    assert restored_bucket == original_bucket

    assert restored_vocabulary.get_id(original_bucket) == vocabulary.get_id(
        original_bucket
    )
