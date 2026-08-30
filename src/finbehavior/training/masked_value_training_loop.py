import torch

from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.models.model import FinBehaviorModel

from .masked_value_training_step import (
    masked_value_training_step,
)
from .masking import MaskedValueExample


def train_masked_values(
    model: FinBehaviorModel,
    prediction_head: MaskedValuePredictionHead,
    optimizer: torch.optim.Optimizer,
    examples: tuple[MaskedValueExample, ...],
    epoch_count: int,
) -> tuple[float, ...]:
    if not examples:
        raise ValueError("Training examples must not be empty")

    if epoch_count <= 0:
        raise ValueError("Epoch count must be positive")

    model.train()
    prediction_head.train()

    epoch_losses = []

    for _ in range(epoch_count):
        total_loss = 0.0

        for example in examples:
            loss = masked_value_training_step(
                model=model,
                prediction_head=prediction_head,
                optimizer=optimizer,
                example=example,
            )

            total_loss += loss.item()

        average_loss = total_loss / len(examples)

        epoch_losses.append(average_loss)

    return tuple(epoch_losses)
