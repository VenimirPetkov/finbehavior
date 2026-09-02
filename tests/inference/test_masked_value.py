import pytest
import torch

from finbehavior.inference.masked_value import (
    decode_top_k_predictions,
)
from finbehavior.tokenization.vocabulary import (
    Vocabulary,
)


def test_decode_top_k_predictions() -> None:
    vocabulary = Vocabulary()

    vocabulary.add_many(
        (
            "alpha",
            "beta",
            "gamma",
        )
    )

    logits = torch.full(
        (len(vocabulary),),
        -10.0,
    )

    logits[vocabulary.get_id("alpha")] = 3.0

    logits[vocabulary.get_id("beta")] = 2.0

    logits[vocabulary.get_id("gamma")] = 1.0

    predictions = decode_top_k_predictions(
        logits=logits,
        vocabulary=vocabulary,
        k=3,
    )

    assert tuple(prediction.token for prediction in predictions) == (
        "alpha",
        "beta",
        "gamma",
    )

    assert (
        predictions[0].probability
        > predictions[1].probability
        > predictions[2].probability
    )


def test_decode_top_k_predictions_rejects_invalid_shape() -> None:
    vocabulary = Vocabulary()

    logits = torch.zeros(
        (
            1,
            len(vocabulary),
        )
    )

    with pytest.raises(
        ValueError,
        match="Logits must have shape",
    ):
        decode_top_k_predictions(
            logits=logits,
            vocabulary=vocabulary,
        )


def test_decode_top_k_predictions_rejects_invalid_k() -> None:
    vocabulary = Vocabulary()

    logits = torch.zeros(len(vocabulary))

    with pytest.raises(
        ValueError,
        match="k must be between",
    ):
        decode_top_k_predictions(
            logits=logits,
            vocabulary=vocabulary,
            k=0,
        )
