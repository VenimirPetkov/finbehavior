from dataclasses import dataclass
from math import isfinite
from pathlib import Path

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
    load_tokenizer,
    save_tokenizer,
)
from finbehavior.tokenization.vocabulary import (
    Vocabulary,
)

CHECKPOINT_VERSION = 1

MODEL_STATE_FILENAME = "model.pt"
TOKENIZER_STATE_FILENAME = "tokenizer.json"


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: FinBehaviorModel
    prediction_head: MaskedValuePredictionHead
    vocabulary: Vocabulary
    bucketizer: QuantileBucketizer
    epoch: int
    validation_loss: float
    top_1_accuracy: float
    top_5_accuracy: float


def save_checkpoint(
    directory: Path,
    model: FinBehaviorModel,
    prediction_head: MaskedValuePredictionHead,
    vocabulary: Vocabulary,
    bucketizer: QuantileBucketizer,
    epoch: int,
    validation_loss: float,
    top_1_accuracy: float,
    top_5_accuracy: float,
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    transformer_block_count: int = DEFAULT_TRANSFORMER_BLOCK_COUNT,
) -> None:
    if epoch < 0:
        raise ValueError("Epoch must not be negative")

    if not isfinite(validation_loss):
        raise ValueError("Validation loss must be finite")

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
        "model_config": {
            "vocabulary_size": len(vocabulary),
            "embedding_dimension": embedding_dimension,
            "transformer_block_count": (transformer_block_count),
        },
        "metrics": {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "top_1_accuracy": top_1_accuracy,
            "top_5_accuracy": top_5_accuracy,
        },
        "model_state_dict": model.state_dict(),
        "prediction_head_state_dict": (prediction_head.state_dict()),
    }

    torch.save(
        state,
        directory / MODEL_STATE_FILENAME,
    )

    save_tokenizer(
        path=directory / TOKENIZER_STATE_FILENAME,
        vocabulary=vocabulary,
        bucketizer=bucketizer,
    )


def load_checkpoint(
    directory: Path,
    device: torch.device,
) -> LoadedCheckpoint:
    vocabulary, bucketizer = load_tokenizer(directory / TOKENIZER_STATE_FILENAME)

    state = torch.load(
        directory / MODEL_STATE_FILENAME,
        map_location=device,
        weights_only=True,
    )

    version = state["version"]

    if version != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {version}")

    model_config = state["model_config"]

    checkpoint_vocabulary_size = model_config["vocabulary_size"]

    if checkpoint_vocabulary_size != len(vocabulary):
        raise ValueError(
            "Checkpoint vocabulary size does not match " "the saved tokenizer"
        )

    embedding_dimension = model_config["embedding_dimension"]

    transformer_block_count = model_config["transformer_block_count"]

    model = build_finbehavior_model(
        vocabulary_size=len(vocabulary),
        embedding_dimension=embedding_dimension,
        transformer_block_count=(transformer_block_count),
    ).to(device)

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=len(vocabulary),
        embedding_dimension=embedding_dimension,
    ).to(device)

    model.load_state_dict(state["model_state_dict"])

    prediction_head.load_state_dict(state["prediction_head_state_dict"])

    metrics = state["metrics"]

    return LoadedCheckpoint(
        model=model,
        prediction_head=prediction_head,
        vocabulary=vocabulary,
        bucketizer=bucketizer,
        epoch=metrics["epoch"],
        validation_loss=metrics["validation_loss"],
        top_1_accuracy=metrics["top_1_accuracy"],
        top_5_accuracy=metrics["top_5_accuracy"],
    )
