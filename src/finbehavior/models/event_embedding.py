import torch
from torch import nn

from finbehavior.tensorization.types import (
    TensorizedEvent,
)

from .config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from .config.event import (
    EVENT_COMPONENT_COUNT,
)
from .field_embedding import FieldEmbedding
from .temporal_projection import TemporalProjection


class EventEmbedding(nn.Module):
    def __init__(
        self,
        field_embedding: FieldEmbedding,
        temporal_projection: TemporalProjection,
        embedding_dimension: int = (DEFAULT_EMBEDDING_DIMENSION),
    ) -> None:
        super().__init__()

        self.field_embedding = field_embedding
        self.temporal_projection = temporal_projection

        self.composition = nn.Linear(
            in_features=(embedding_dimension * EVENT_COMPONENT_COUNT),
            out_features=embedding_dimension,
        )

    def forward(
        self,
        event: TensorizedEvent,
    ) -> torch.Tensor:
        event_vector = self.field_embedding.token_embedding(event.event_token_id)

        if event.key_ids.numel() == 0:
            fields_vector = torch.zeros_like(event_vector)
        else:
            field_vectors = self.field_embedding(
                key_ids=event.key_ids,
                value_ids=event.value_ids,
            )

            fields_vector = field_vectors.mean(dim=0)

        temporal_vector = self.temporal_projection(
            calendar_features=(event.calendar_features),
            elapsed_time_feature=(event.elapsed_time_feature),
        )

        combined = torch.cat(
            (
                event_vector,
                fields_vector,
                temporal_vector,
            ),
            dim=0,
        )

        return self.composition(combined)
