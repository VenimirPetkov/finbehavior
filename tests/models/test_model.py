import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.encoder import (
    TransformerEncoder,
)
from finbehavior.models.event_embedding import (
    EventEmbedding,
)
from finbehavior.models.field_embedding import (
    FieldEmbedding,
)
from finbehavior.models.history_embedding import (
    HistoryEmbedding,
)
from finbehavior.models.model import (
    FinBehaviorModel,
)
from finbehavior.models.profile_embedding import (
    ProfileEmbedding,
)
from finbehavior.models.temporal_projection import (
    TemporalProjection,
)
from finbehavior.models.user_sequence_embedding import (
    UserSequenceEmbedding,
)
from finbehavior.tensorization.types import (
    TensorizedEvent,
    TensorizedProfile,
    TensorizedUser,
)
from finbehavior.tokenization.special_tokens import (
    EVT_TOKEN,
    USR_TOKEN,
)
from finbehavior.tokenization.vocabulary import (
    build_vocabulary,
)


def test_finbehavior_model_encodes_user(
    transformer_block_factory,
):
    vocabulary = build_vocabulary()

    field_embedding = FieldEmbedding(
        vocabulary_size=len(vocabulary),
    )

    profile_embedding = ProfileEmbedding(
        field_embedding=field_embedding,
    )

    temporal_projection = TemporalProjection()

    event_embedding = EventEmbedding(
        field_embedding=field_embedding,
        temporal_projection=temporal_projection,
    )

    history_embedding = HistoryEmbedding(
        event_embedding=event_embedding,
    )

    user_sequence_embedding = UserSequenceEmbedding(
        profile_embedding=profile_embedding,
        history_embedding=history_embedding,
    )

    encoder = TransformerEncoder(
        blocks=(
            transformer_block_factory(),
            transformer_block_factory(),
        ),
    )

    model = FinBehaviorModel(
        user_sequence_embedding=user_sequence_embedding,
        encoder=encoder,
    )

    profile = TensorizedProfile(
        user_token_id=torch.tensor(
            vocabulary.get_id(USR_TOKEN),
            dtype=torch.long,
        ),
        key_ids=torch.empty(
            0,
            dtype=torch.long,
        ),
        value_ids=torch.empty(
            0,
            dtype=torch.long,
        ),
    )

    user = TensorizedUser(
        user_id=1,
        profile=profile,
        events=(),
    )

    encoded_sequence = model.encode_sequence(user)

    user_representation = model(user)

    assert encoded_sequence.shape == (
        1,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    assert user_representation.shape == (DEFAULT_EMBEDDING_DIMENSION,)

    assert torch.allclose(
        user_representation,
        encoded_sequence[0],
    )


def test_finbehavior_model_encodes_users_with_different_sequence_lengths(
    transformer_block_factory,
):
    vocabulary = build_vocabulary()

    field_embedding = FieldEmbedding(
        vocabulary_size=len(vocabulary),
    )

    profile_embedding = ProfileEmbedding(
        field_embedding=field_embedding,
    )

    temporal_projection = TemporalProjection()

    event_embedding = EventEmbedding(
        field_embedding=field_embedding,
        temporal_projection=temporal_projection,
    )

    history_embedding = HistoryEmbedding(
        event_embedding=event_embedding,
    )

    user_sequence_embedding = UserSequenceEmbedding(
        profile_embedding=profile_embedding,
        history_embedding=history_embedding,
    )

    encoder = TransformerEncoder(
        blocks=(
            transformer_block_factory(),
            transformer_block_factory(),
        ),
    )

    model = FinBehaviorModel(
        user_sequence_embedding=user_sequence_embedding,
        encoder=encoder,
    )

    profile = TensorizedProfile(
        user_token_id=torch.tensor(
            vocabulary.get_id(USR_TOKEN),
            dtype=torch.long,
        ),
        key_ids=torch.empty(
            0,
            dtype=torch.long,
        ),
        value_ids=torch.empty(
            0,
            dtype=torch.long,
        ),
    )

    def create_event() -> TensorizedEvent:
        return TensorizedEvent(
            event_token_id=torch.tensor(
                vocabulary.get_id(EVT_TOKEN),
                dtype=torch.long,
            ),
            key_ids=torch.empty(
                0,
                dtype=torch.long,
            ),
            value_ids=torch.empty(
                0,
                dtype=torch.long,
            ),
            calendar_features=torch.zeros(
                6,
                dtype=torch.float32,
            ),
            elapsed_time_feature=torch.tensor(
                0.0,
                dtype=torch.float32,
            ),
        )

    user_a = TensorizedUser(
        user_id=1,
        profile=profile,
        events=(create_event(),),
    )

    user_b = TensorizedUser(
        user_id=2,
        profile=profile,
        events=(
            create_event(),
            create_event(),
            create_event(),
        ),
    )

    encoded_batch = model.encode_users(
        (
            user_a,
            user_b,
        )
    )

    assert encoded_batch.shape == (
        2,
        4,
        DEFAULT_EMBEDDING_DIMENSION,
    )

    encoded_user_a = model.encode_sequence(user_a)

    encoded_user_b = model.encode_sequence(user_b)

    assert torch.allclose(
        encoded_batch[
            0,
            : encoded_user_a.shape[0],
        ],
        encoded_user_a,
        atol=1e-5,
    )

    assert torch.allclose(
        encoded_batch[
            1,
            : encoded_user_b.shape[0],
        ],
        encoded_user_b,
        atol=1e-5,
    )
