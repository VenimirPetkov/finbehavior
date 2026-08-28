import torch
from torch import nn

from finbehavior.tokenization.config.temporal import (
    CALENDAR_FEATURE_DIMENSION,
)

from .config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)

ELAPSED_TIME_FEATURE_DIMENSION = 1
TEMPORAL_FEATURE_DIMENSION = CALENDAR_FEATURE_DIMENSION + ELAPSED_TIME_FEATURE_DIMENSION


class TemporalProjection(nn.Module):
    def __init__(
        self,
        embedding_dimension: int = (DEFAULT_EMBEDDING_DIMENSION),
    ) -> None:
        super().__init__()

        self.projection = nn.Linear(
            in_features=TEMPORAL_FEATURE_DIMENSION,
            out_features=embedding_dimension,
        )

    def forward(
        self,
        calendar_features: torch.Tensor,
        elapsed_time_feature: torch.Tensor,
    ) -> torch.Tensor:
        temporal_features = torch.cat(
            (
                calendar_features,
                elapsed_time_feature.unsqueeze(0),
            )
        )

        return self.projection(temporal_features)
