from pathlib import Path

import torch

from finbehavior.models.factory import (
    build_finbehavior_model,
)
from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.persistence.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from finbehavior.tokenization.numerical import (
    QuantileBucketizer,
)
from finbehavior.tokenization.vocabulary import (
    Vocabulary,
)


def test_checkpoint_round_trip(
    tmp_path: Path,
):
    torch.manual_seed(42)

    vocabulary = Vocabulary()

    vocabulary.add_many(
        (
            "amount",
            "merchant",
        )
    )

    bucketizer = QuantileBucketizer(
        number_of_buckets=2,
    )

    bucketizer.fit(
        key_token="amount",
        values=(
            10.0,
            20.0,
            30.0,
            40.0,
        ),
    )

    vocabulary.add_many(bucketizer.get_bucket_tokens("amount"))

    model = build_finbehavior_model(
        vocabulary_size=len(vocabulary),
    )

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=len(vocabulary),
    )

    checkpoint_directory = tmp_path / "checkpoint"

    save_checkpoint(
        directory=checkpoint_directory,
        model=model,
        prediction_head=prediction_head,
        vocabulary=vocabulary,
        bucketizer=bucketizer,
        epoch=5,
        validation_loss=2.4135,
        top_1_accuracy=0.29,
        top_5_accuracy=0.624,
    )

    loaded = load_checkpoint(
        directory=checkpoint_directory,
        device=torch.device("cpu"),
    )

    assert loaded.epoch == 5
    assert loaded.validation_loss == 2.4135
    assert loaded.top_1_accuracy == 0.29
    assert loaded.top_5_accuracy == 0.624

    assert loaded.vocabulary.get_tokens() == vocabulary.get_tokens()

    assert loaded.bucketizer.get_all_boundaries() == bucketizer.get_all_boundaries()

    original_model_state = model.state_dict()

    loaded_model_state = loaded.model.state_dict()

    assert original_model_state.keys() == loaded_model_state.keys()

    for name in original_model_state:
        assert torch.equal(
            original_model_state[name],
            loaded_model_state[name],
        )

    original_head_state = prediction_head.state_dict()

    loaded_head_state = loaded.prediction_head.state_dict()

    assert original_head_state.keys() == loaded_head_state.keys()

    for name in original_head_state:
        assert torch.equal(
            original_head_state[name],
            loaded_head_state[name],
        )
