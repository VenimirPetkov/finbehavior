from pathlib import Path

import torch

from finbehavior.models.factory import (
    build_finbehavior_model,
)
from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.persistence.checkpoint import (
    CHECKPOINT_VERSION,
    MODEL_STATE_FILENAME,
    checkpoint_exists,
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
) -> None:
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
        allowed_token_ids_by_key_id={
            vocabulary.get_id("amount"): tuple(
                vocabulary.get_id(token)
                for token in bucketizer.get_bucket_tokens("amount")
            ),
        },
    )

    parameters = list(model.parameters()) + list(prediction_head.parameters())

    optimizer = torch.optim.AdamW(
        parameters,
        lr=0.003,
    )

    loss = sum(parameter.square().mean() for parameter in parameters)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    original_optimizer_state = optimizer.state_dict()

    assert original_optimizer_state["state"]

    checkpoint_directory = tmp_path / "checkpoint"

    assert not checkpoint_exists(checkpoint_directory)

    save_checkpoint(
        directory=checkpoint_directory,
        model=model,
        prediction_head=prediction_head,
        optimizer=optimizer,
        vocabulary=vocabulary,
        bucketizer=bucketizer,
        epoch=5,
        validation_loss=2.4135,
        top_1_accuracy=0.29,
        top_5_accuracy=0.624,
        best_validation_epoch=5,
        best_validation_loss=2.4135,
    )

    assert checkpoint_exists(checkpoint_directory)

    saved_state = torch.load(
        checkpoint_directory / MODEL_STATE_FILENAME,
        map_location="cpu",
        weights_only=True,
    )

    assert saved_state["version"] == CHECKPOINT_VERSION == 3
    assert saved_state["tokenizer_state"]["vocabulary"] == list(vocabulary.get_tokens())
    assert not tuple(checkpoint_directory.glob("*.tmp"))

    # model.pt is the atomic, authoritative restore unit. The JSON sidecar is
    # a human-readable export and must not be required for recovery.
    (checkpoint_directory / "tokenizer.json").unlink()
    assert checkpoint_exists(checkpoint_directory)

    loaded = load_checkpoint(
        directory=checkpoint_directory,
        device=torch.device("cpu"),
    )

    assert loaded.epoch == 5
    assert loaded.validation_loss == 2.4135
    assert loaded.top_1_accuracy == 0.29
    assert loaded.top_5_accuracy == 0.624

    assert loaded.best_validation_epoch == 5

    assert loaded.best_validation_loss == 2.4135

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

    assert torch.equal(
        prediction_head.allowed_value_mask,
        loaded.prediction_head.allowed_value_mask,
    )

    loaded_optimizer_state = loaded.optimizer_state_dict

    assert (
        original_optimizer_state["param_groups"]
        == loaded_optimizer_state["param_groups"]
    )

    assert (
        original_optimizer_state["state"].keys()
        == loaded_optimizer_state["state"].keys()
    )

    for parameter_id, original_state in original_optimizer_state["state"].items():
        loaded_state = loaded_optimizer_state["state"][parameter_id]

        assert original_state.keys() == loaded_state.keys()

        for key, original_value in original_state.items():
            loaded_value = loaded_state[key]

            if torch.is_tensor(original_value):
                assert torch.equal(
                    original_value,
                    loaded_value,
                )
            else:
                assert original_value == loaded_value
