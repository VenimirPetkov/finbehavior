from dataclasses import dataclass

import torch

from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.models.model import FinBehaviorModel

from .masking import MaskedValueExample

EVENT_SEQUENCE_OFFSET = 1


@dataclass(frozen=True)
class MaskedValueBatchPrediction:
    logits: torch.Tensor
    target_token_ids: torch.Tensor


def predict_masked_value_batch(
    model: FinBehaviorModel,
    prediction_head: MaskedValuePredictionHead,
    examples: tuple[MaskedValueExample, ...],
) -> MaskedValueBatchPrediction:
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

    return MaskedValueBatchPrediction(
        logits=logits,
        target_token_ids=target_token_ids,
    )
