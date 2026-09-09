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
    torch.manual_seed(0)

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

    key_value_pairs = torch.cat(
        (
            embedding.token_embedding(key_ids),
            embedding.token_embedding(value_ids),
        ),
        dim=-1,
    )

    assert torch.allclose(
        field_vectors,
        embedding.pair_encoder(key_value_pairs),
    )


def test_field_embedding_preserves_key_value_bindings_after_pooling():
    torch.manual_seed(0)

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

    original = embedding(
        key_ids=key_ids,
        value_ids=value_ids,
    ).mean(dim=0)

    swapped = embedding(
        key_ids=key_ids,
        value_ids=value_ids.flip(0),
    ).mean(dim=0)

    assert not torch.allclose(
        original,
        swapped,
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
