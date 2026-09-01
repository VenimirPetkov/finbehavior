from dataclasses import dataclass

import torch

from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.models.model import (
    FinBehaviorModel,
)
from finbehavior.training.config.batch import (
    DEFAULT_BATCH_SIZE,
)
from finbehavior.training.example_batching import (
    build_length_aware_batches,
)
from finbehavior.training.masked_value_batch_prediction import (
    predict_masked_value_batch,
)
from finbehavior.training.masked_value_loss import (
    masked_value_loss,
)
from finbehavior.training.masking import (
    MaskedValueExample,
)

from .top_k_accuracy import (
    top_k_accuracy,
)


@dataclass(frozen=True)
class MaskedValueMetrics:
    loss: float
    top_1_accuracy: float
    top_5_accuracy: float


def evaluate_masked_value_metrics(
    model: FinBehaviorModel,
    prediction_head: MaskedValuePredictionHead,
    examples: tuple[MaskedValueExample, ...],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> MaskedValueMetrics:
    if not examples:
        raise ValueError("Evaluation examples must not be empty")

    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    model_was_training = model.training

    prediction_head_was_training = prediction_head.training

    model.eval()
    prediction_head.eval()

    batches = build_length_aware_batches(
        examples=examples,
        batch_size=batch_size,
    )

    total_loss = 0.0
    total_top_1_correct = 0.0
    total_top_5_correct = 0.0

    try:
        with torch.no_grad():
            for batch in batches:
                prediction = predict_masked_value_batch(
                    model=model,
                    prediction_head=prediction_head,
                    examples=batch,
                )

                loss = masked_value_loss(
                    prediction.logits,
                    prediction.target_token_ids,
                )

                top_1 = top_k_accuracy(
                    logits=prediction.logits,
                    target_token_ids=(prediction.target_token_ids),
                    k=1,
                )

                top_5 = top_k_accuracy(
                    logits=prediction.logits,
                    target_token_ids=(prediction.target_token_ids),
                    k=min(
                        5,
                        prediction.logits.shape[1],
                    ),
                )

                batch_size_actual = len(batch)

                total_loss += loss.item() * batch_size_actual

                total_top_1_correct += top_1 * batch_size_actual

                total_top_5_correct += top_5 * batch_size_actual

    finally:
        model.train(model_was_training)

        prediction_head.train(prediction_head_was_training)

    example_count = len(examples)

    return MaskedValueMetrics(
        loss=(total_loss / example_count),
        top_1_accuracy=(total_top_1_correct / example_count),
        top_5_accuracy=(total_top_5_correct / example_count),
    )
