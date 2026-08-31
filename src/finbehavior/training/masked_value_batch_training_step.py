import torch

from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.models.model import FinBehaviorModel

from .masked_value_batch_training_loss import (
    masked_value_batch_training_loss,
)
from .masking import MaskedValueExample


def masked_value_batch_training_step(
    model: FinBehaviorModel,
    prediction_head: MaskedValuePredictionHead,
    optimizer: torch.optim.Optimizer,
    examples: tuple[MaskedValueExample, ...],
) -> torch.Tensor:
    optimizer.zero_grad()

    loss = masked_value_batch_training_loss(
        model=model,
        prediction_head=prediction_head,
        examples=examples,
    )

    loss.backward()

    optimizer.step()

    return loss.detach()
