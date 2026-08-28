import pytest
import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.field_embedding import (
    FieldEmbedding,
)
from finbehavior.tokenization.vocabulary import (
    build_vocabulary,
)


def test_field_embedding():
    vocabulary = build_vocabulary()

    embedding = FieldEmbedding(
        vocabulary_size=len(vocabulary),
    )

    key_ids = torch.tensor(
        [23, 24],
        dtype=torch.long,
    )

    value_ids = torch.tensor(
        [113, 63],
        dtype=torch.long,
    )

    field_vectors = embedding(
        key_ids=key_ids,
        value_ids=value_ids,
    )

    assert field_vectors.shape == (
        2,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    key_vectors = embedding.token_embedding(key_ids)

    value_vectors = embedding.token_embedding(value_ids)

    assert torch.allclose(
        field_vectors,
        key_vectors + value_vectors,
    )


def test_field_embedding_rejects_mismatched_shapes():
    vocabulary = build_vocabulary()

    embedding = FieldEmbedding(
        vocabulary_size=len(vocabulary),
    )

    key_ids = torch.tensor(
        [23, 24],
        dtype=torch.long,
    )

    value_ids = torch.tensor(
        [113],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="matching shapes",
    ):
        embedding(
            key_ids=key_ids,
            value_ids=value_ids,
        )
