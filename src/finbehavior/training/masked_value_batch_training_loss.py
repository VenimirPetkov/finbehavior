import torch

from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.models.model import FinBehaviorModel

from .masked_value_loss import masked_value_loss
from .masking import MaskedValueExample

EVENT_SEQUENCE_OFFSET = 1


def masked_value_batch_training_loss(
    model: FinBehaviorModel,
    prediction_head: MaskedValuePredictionHead,
    examples: tuple[MaskedValueExample, ...],
) -> torch.Tensor:
    if not examples:
        raise ValueError("Batch examples must not be empty")

    users = tuple(example.user for example in examples)

    encoded_sequences = model.encode_users(users)

    device = encoded_sequences.device

    batch_indices = torch.arange(
        len(examples),
        device=device,
    )

    event_sequence_indices = torch.tensor(
        [example.event_index + EVENT_SEQUENCE_OFFSET for example in examples],
        dtype=torch.long,
        device=device,
    )

    event_representations = encoded_sequences[
        batch_indices,
        event_sequence_indices,
    ]

    logits = prediction_head(event_representations)

    target_token_ids = torch.stack(
        tuple(example.target_token_id.to(device) for example in examples)
    )

    return masked_value_loss(
        logits,
        target_token_ids,
    )
