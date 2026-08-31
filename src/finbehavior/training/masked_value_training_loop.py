import torch

from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.models.model import FinBehaviorModel

from .config.batch import (
    DEFAULT_BATCH_SHUFFLE_SEED,
    DEFAULT_BATCH_SIZE,
)
from .example_batching import (
    build_length_aware_batches,
)
from .masked_value_batch_training_step import (
    masked_value_batch_training_step,
)
from .masking import MaskedValueExample


def train_masked_values(
    model: FinBehaviorModel,
    prediction_head: MaskedValuePredictionHead,
    optimizer: torch.optim.Optimizer,
    examples: tuple[MaskedValueExample, ...],
    epoch_count: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle_seed: int = DEFAULT_BATCH_SHUFFLE_SEED,
) -> tuple[float, ...]:
    if not examples:
        raise ValueError("Training examples must not be empty")

    if epoch_count <= 0:
        raise ValueError("Epoch count must be positive")

    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    model.train()
    prediction_head.train()

    epoch_losses = []

    for epoch_index in range(epoch_count):
        batches = build_length_aware_batches(
            examples=examples,
            batch_size=batch_size,
            shuffle=True,
            seed=(shuffle_seed + epoch_index),
        )

        total_loss = 0.0

        for batch in batches:
            loss = masked_value_batch_training_step(
                model=model,
                prediction_head=prediction_head,
                optimizer=optimizer,
                examples=batch,
            )

            total_loss += loss.item() * len(batch)

        average_loss = total_loss / len(examples)

        epoch_losses.append(average_loss)

    return tuple(epoch_losses)
