import torch
from torch import nn

from finbehavior.tensorization.types import (
    TensorizedEvent,
)

from .event_embedding import EventEmbedding


class HistoryEmbedding(nn.Module):
    def __init__(
        self,
        event_embedding: EventEmbedding,
    ) -> None:
        super().__init__()

        self.event_embedding = event_embedding

    def forward(
        self,
        events: tuple[TensorizedEvent, ...],
    ) -> torch.Tensor:
        if not events:
            return self.event_embedding.composition.weight.new_empty(
                (
                    0,
                    self.event_embedding.composition.out_features,
                )
            )

        event_vectors = tuple(self.event_embedding(event) for event in events)

        return torch.stack(
            event_vectors,
            dim=0,
        )
