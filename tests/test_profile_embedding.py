import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.field_embedding import (
    FieldEmbedding,
)
from finbehavior.models.profile_embedding import (
    ProfileEmbedding,
)
from finbehavior.tensorization.types import (
    TensorizedProfile,
)
from finbehavior.tokenization.vocabulary import (
    build_vocabulary,
)


def test_profile_embedding():
    vocabulary = build_vocabulary()

    field_embedding = FieldEmbedding(
        vocabulary_size=len(vocabulary),
    )

    profile_embedding = ProfileEmbedding(
        field_embedding=field_embedding,
    )

    profile = TensorizedProfile(
        user_token_id=torch.tensor(
            2,
            dtype=torch.long,
        ),
        key_ids=torch.tensor(
            [23, 24],
            dtype=torch.long,
        ),
        value_ids=torch.tensor(
            [113, 63],
            dtype=torch.long,
        ),
    )

    profile_vector = profile_embedding(profile)

    user_vector = field_embedding.token_embedding(profile.user_token_id)

    field_vectors = field_embedding(
        key_ids=profile.key_ids,
        value_ids=profile.value_ids,
    )

    expected = user_vector + field_vectors.mean(dim=0)

    assert profile_vector.shape == (DEFAULT_EMBEDDING_DIMENSION,)

    assert torch.allclose(
        profile_vector,
        expected,
    )


def test_profile_embedding_without_fields():
    vocabulary = build_vocabulary()

    field_embedding = FieldEmbedding(
        vocabulary_size=len(vocabulary),
    )

    profile_embedding = ProfileEmbedding(
        field_embedding=field_embedding,
    )

    profile = TensorizedProfile(
        user_token_id=torch.tensor(
            2,
            dtype=torch.long,
        ),
        key_ids=torch.tensor(
            [],
            dtype=torch.long,
        ),
        value_ids=torch.tensor(
            [],
            dtype=torch.long,
        ),
    )

    profile_vector = profile_embedding(profile)

    user_vector = field_embedding.token_embedding(profile.user_token_id)

    assert torch.allclose(
        profile_vector,
        user_vector,
    )
