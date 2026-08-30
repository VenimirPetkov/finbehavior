import pytest

from finbehavior.models.config.encoder import (
    DEFAULT_TRANSFORMER_BLOCK_COUNT,
)
from finbehavior.models.factory import (
    build_finbehavior_model,
)
from finbehavior.models.model import (
    FinBehaviorModel,
)


def test_model_factory_builds_finbehavior_model():
    model = build_finbehavior_model(
        vocabulary_size=100,
    )

    assert isinstance(
        model,
        FinBehaviorModel,
    )

    assert len(model.encoder.blocks) == DEFAULT_TRANSFORMER_BLOCK_COUNT


def test_model_factory_shares_field_embedding():
    model = build_finbehavior_model(
        vocabulary_size=100,
    )

    profile_field_embedding = (
        model.user_sequence_embedding.profile_embedding.field_embedding
    )

    event_field_embedding = (
        model.user_sequence_embedding.history_embedding.event_embedding.field_embedding
    )

    assert profile_field_embedding is event_field_embedding


def test_model_factory_builds_independent_transformer_blocks():
    model = build_finbehavior_model(
        vocabulary_size=100,
        transformer_block_count=2,
    )

    assert model.encoder.blocks[0] is not model.encoder.blocks[1]


def test_model_factory_rejects_invalid_block_count():
    with pytest.raises(
        ValueError,
        match="Transformer block count must be positive",
    ):
        build_finbehavior_model(
            vocabulary_size=100,
            transformer_block_count=0,
        )
