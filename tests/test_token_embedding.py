import torch

from finbehavior.models.config.embedding import DEFAULT_EMBEDDING_DIMENSION
from finbehavior.models.embedding import (
    TokenEmbedding,
)
from finbehavior.tokenization.vocabulary import (
    build_vocabulary,
)


def test_token_embedding():
    vocabulary = build_vocabulary()

    embedding = TokenEmbedding(
        vocabulary_size=len(vocabulary),
    )

    token_ids = torch.tensor(
        [23, 24, 23],
        dtype=torch.long,
    )

    vectors = embedding(token_ids)

    assert vectors.shape == (
        3,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    assert torch.equal(
        vectors[0],
        vectors[2],
    )

    assert embedding.embedding.weight.requires_grad is True
