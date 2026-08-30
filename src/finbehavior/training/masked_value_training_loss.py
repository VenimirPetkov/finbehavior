import torch

from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.models.model import FinBehaviorModel

from .masked_value_loss import masked_value_loss
from .masking import MaskedValueExample

EVENT_SEQUENCE_OFFSET = 1


def masked_value_training_loss(
    model: FinBehaviorModel,
    prediction_head: MaskedValuePredictionHead,
    example: MaskedValueExample,
) -> torch.Tensor:
    encoded_sequence = model.encode_sequence(example.user)

    event_sequence_index = example.event_index + EVENT_SEQUENCE_OFFSET

    event_representation = encoded_sequence[event_sequence_index]

    logits = prediction_head(event_representation)

    return masked_value_loss(
        logits,
        example.target_token_id,
    )
