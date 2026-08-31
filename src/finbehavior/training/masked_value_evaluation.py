import torch

from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.models.model import FinBehaviorModel

from .config.batch import DEFAULT_BATCH_SIZE
from .masked_value_batch_training_loss import (
    masked_value_batch_training_loss,
)
from .masking import MaskedValueExample


def evaluate_masked_values(
    model: FinBehaviorModel,
    prediction_head: MaskedValuePredictionHead,
    examples: tuple[MaskedValueExample, ...],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> float:
    if not examples:
        raise ValueError("Evaluation examples must not be empty")

    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    model_was_training = model.training
    prediction_head_was_training = prediction_head.training

    model.eval()
    prediction_head.eval()

    total_loss = 0.0

    try:
        with torch.no_grad():
            for start_index in range(
                0,
                len(examples),
                batch_size,
            ):
                batch = examples[start_index : start_index + batch_size]

                loss = masked_value_batch_training_loss(
                    model=model,
                    prediction_head=prediction_head,
                    examples=batch,
                )

                total_loss += loss.item() * len(batch)

    finally:
        model.train(model_was_training)

        prediction_head.train(prediction_head_was_training)

    return total_loss / len(examples)
