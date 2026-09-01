import torch

from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.models.model import FinBehaviorModel

from .masked_value_batch_prediction import (
    predict_masked_value_batch,
)
from .masked_value_loss import (
    masked_value_loss,
)
from .masking import MaskedValueExample


def masked_value_batch_training_loss(
    model: FinBehaviorModel,
    prediction_head: MaskedValuePredictionHead,
    examples: tuple[MaskedValueExample, ...],
) -> torch.Tensor:
    prediction = predict_masked_value_batch(
        model=model,
        prediction_head=prediction_head,
        examples=examples,
    )

    return masked_value_loss(
        prediction.logits,
        prediction.target_token_ids,
    )
