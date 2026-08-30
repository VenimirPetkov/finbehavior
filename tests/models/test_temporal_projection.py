import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.temporal_projection import (
    TEMPORAL_FEATURE_DIMENSION,
    TemporalProjection,
)


def test_temporal_projection():
    projection = TemporalProjection()

    calendar_features = torch.tensor(
        [
            0.5,
            -0.5,
            0.25,
            -0.25,
            1.0,
            0.0,
        ],
        dtype=torch.float32,
    )

    elapsed_time_feature = torch.tensor(
        17.42,
        dtype=torch.float32,
    )

    temporal_vector = projection(
        calendar_features=calendar_features,
        elapsed_time_feature=elapsed_time_feature,
    )

    assert temporal_vector.shape == (DEFAULT_EMBEDDING_DIMENSION,)

    assert projection.projection.in_features == (TEMPORAL_FEATURE_DIMENSION)

    assert projection.projection.out_features == (DEFAULT_EMBEDDING_DIMENSION)

    assert projection.projection.weight.requires_grad is True
