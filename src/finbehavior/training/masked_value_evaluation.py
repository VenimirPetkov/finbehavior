import torch

from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.models.model import FinBehaviorModel

from .masked_value_training_loss import (
    masked_value_training_loss,
)
from .masking import MaskedValueExample


def evaluate_masked_values(
    model: FinBehaviorModel,
    prediction_head: MaskedValuePredictionHead,
    examples: tuple[MaskedValueExample, ...],
) -> float:
    if not examples:
        raise ValueError("Evaluation examples must not be empty")

    model_was_training = model.training
    prediction_head_was_training = prediction_head.training

    model.eval()
    prediction_head.eval()

    total_loss = 0.0

    with torch.no_grad():
        for example in examples:
            loss = masked_value_training_loss(
                model=model,
                prediction_head=prediction_head,
                example=example,
            )

            total_loss += loss.item()

    model.train(model_was_training)
    prediction_head.train(prediction_head_was_training)

    return total_loss / len(examples)
