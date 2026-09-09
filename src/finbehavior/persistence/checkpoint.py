import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.config.encoder import (
    DEFAULT_TRANSFORMER_BLOCK_COUNT,
)
from finbehavior.models.factory import (
    build_finbehavior_model,
)
from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.models.model import (
    FinBehaviorModel,
)
from finbehavior.tokenization.numerical import (
    QuantileBucketizer,
)
from finbehavior.tokenization.persistence import (
    save_tokenizer,
    tokenizer_from_state,
    tokenizer_to_state,
)
from finbehavior.tokenization.vocabulary import (
    Vocabulary,
)

CHECKPOINT_VERSION = 3

MODEL_STATE_FILENAME = "model.pt"
TOKENIZER_STATE_FILENAME = "tokenizer.json"


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: FinBehaviorModel
    prediction_head: MaskedValuePredictionHead
    vocabulary: Vocabulary
    bucketizer: QuantileBucketizer

    optimizer_state_dict: dict[str, Any]

    epoch: int

    validation_loss: float
    top_1_accuracy: float
    top_5_accuracy: float

    best_validation_epoch: int
    best_validation_loss: float


def checkpoint_exists(
    directory: Path,
) -> bool:
    return (directory / MODEL_STATE_FILENAME).is_file()


def save_checkpoint(
    directory: Path,
    model: FinBehaviorModel,
    prediction_head: MaskedValuePredictionHead,
    optimizer: torch.optim.Optimizer,
    vocabulary: Vocabulary,
    bucketizer: QuantileBucketizer,
    epoch: int,
    validation_loss: float,
    top_1_accuracy: float,
    top_5_accuracy: float,
    best_validation_epoch: int,
    best_validation_loss: float,
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    transformer_block_count: int = DEFAULT_TRANSFORMER_BLOCK_COUNT,
) -> None:
    if epoch < 0:
        raise ValueError("Epoch must not be negative")

    if best_validation_epoch < 0:
        raise ValueError("Best validation epoch must not be negative")

    if best_validation_epoch > epoch:
        raise ValueError("Best validation epoch cannot be after current epoch")

    if not isfinite(validation_loss):
        raise ValueError("Validation loss must be finite")

    if not isfinite(best_validation_loss):
        raise ValueError("Best validation loss must be finite")

    if not 0.0 <= top_1_accuracy <= 1.0:
        raise ValueError("Top-1 accuracy must be between 0 and 1")

    if not 0.0 <= top_5_accuracy <= 1.0:
        raise ValueError("Top-5 accuracy must be between 0 and 1")

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        "version": CHECKPOINT_VERSION,
        "tokenizer_state": tokenizer_to_state(vocabulary, bucketizer),
        "model_config": {
            "vocabulary_size": len(vocabulary),
            "embedding_dimension": (embedding_dimension),
            "transformer_block_count": (transformer_block_count),
            "allowed_token_ids_by_key_id": (
                prediction_head.get_allowed_token_ids_by_key_id()
            ),
        },
        "training_state": {
            "epoch": epoch,
            "best_validation_epoch": (best_validation_epoch),
            "best_validation_loss": (best_validation_loss),
        },
        "metrics": {
            "validation_loss": validation_loss,
            "top_1_accuracy": top_1_accuracy,
            "top_5_accuracy": top_5_accuracy,
        },
        "model_state_dict": (model.state_dict()),
        "prediction_head_state_dict": (prediction_head.state_dict()),
        "optimizer_state_dict": (optimizer.state_dict()),
    }

    model_path = directory / MODEL_STATE_FILENAME
    tokenizer_path = directory / TOKENIZER_STATE_FILENAME

    unique_suffix = uuid4().hex
    temporary_model_path = directory / f".{MODEL_STATE_FILENAME}.{unique_suffix}.tmp"
    temporary_tokenizer_path = (
        directory / f".{TOKENIZER_STATE_FILENAME}.{unique_suffix}.tmp"
    )

    try:
        torch.save(
            state,
            temporary_model_path,
        )

        save_tokenizer(
            path=temporary_tokenizer_path,
            vocabulary=vocabulary,
            bucketizer=bucketizer,
        )

        os.replace(temporary_tokenizer_path, tokenizer_path)
        os.replace(temporary_model_path, model_path)
    finally:
        temporary_model_path.unlink(missing_ok=True)
        temporary_tokenizer_path.unlink(missing_ok=True)


def load_checkpoint(
    directory: Path,
    device: torch.device,
) -> LoadedCheckpoint:
    state = torch.load(
        directory / MODEL_STATE_FILENAME,
        map_location=device,
        weights_only=True,
    )

    version = state["version"]

    if version != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: " f"{version}")

    vocabulary, bucketizer = tokenizer_from_state(state["tokenizer_state"])

    model_config = state["model_config"]

    checkpoint_vocabulary_size = model_config["vocabulary_size"]

    if checkpoint_vocabulary_size != len(vocabulary):
        raise ValueError(
            "Checkpoint vocabulary size does not " "match the saved tokenizer"
        )

    embedding_dimension = model_config["embedding_dimension"]

    transformer_block_count = model_config["transformer_block_count"]

    model = build_finbehavior_model(
        vocabulary_size=len(vocabulary),
        embedding_dimension=(embedding_dimension),
        transformer_block_count=(transformer_block_count),
    ).to(device)

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=len(vocabulary),
        embedding_dimension=(embedding_dimension),
        allowed_token_ids_by_key_id=(model_config["allowed_token_ids_by_key_id"]),
    ).to(device)

    model.load_state_dict(state["model_state_dict"])

    prediction_head.load_state_dict(state["prediction_head_state_dict"])

    training_state = state["training_state"]

    metrics = state["metrics"]

    return LoadedCheckpoint(
        model=model,
        prediction_head=prediction_head,
        vocabulary=vocabulary,
        bucketizer=bucketizer,
        optimizer_state_dict=state["optimizer_state_dict"],
        epoch=training_state["epoch"],
        validation_loss=metrics["validation_loss"],
        top_1_accuracy=metrics["top_1_accuracy"],
        top_5_accuracy=metrics["top_5_accuracy"],
        best_validation_epoch=(training_state["best_validation_epoch"]),
        best_validation_loss=(training_state["best_validation_loss"]),
    )
