import torch

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
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
from finbehavior.tokenization.config.temporal import (
    CALENDAR_FEATURE_DIMENSION,
)
from finbehavior.tokenization.special_tokens import (
    EVT_TOKEN,
    USR_TOKEN,
)
from finbehavior.tokenization.vocabulary import (
    build_vocabulary,
)


def build_user_sequence_embedding():
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

    return vocabulary, user_sequence_embedding


def test_user_sequence_embedding():
    vocabulary, user_sequence_embedding = build_user_sequence_embedding()

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

    event = TensorizedEvent(
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
            CALENDAR_FEATURE_DIMENSION,
            dtype=torch.float32,
        ),
        elapsed_time_feature=torch.tensor(
            0.0,
            dtype=torch.float32,
        ),
    )

    user = TensorizedUser(
        user_id=1,
        profile=profile,
        events=(event,),
    )

    sequence = user_sequence_embedding(user)

    assert sequence.shape == (
        2,
        DEFAULT_EMBEDDING_DIMENSION,
    )


def test_user_sequence_embedding_supports_profile_only_user():
    vocabulary, user_sequence_embedding = build_user_sequence_embedding()

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

    sequence = user_sequence_embedding(user)

    assert sequence.shape == (
        1,
        DEFAULT_EMBEDDING_DIMENSION,
    )
