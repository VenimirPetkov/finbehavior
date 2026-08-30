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
from finbehavior.models.feed_forward import (
    FeedForward,
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
from finbehavior.models.self_attention import (
    SelfAttention,
)
from finbehavior.models.self_attention_block import (
    SelfAttentionBlock,
)
from finbehavior.models.temporal_projection import (
    TemporalProjection,
)
from finbehavior.models.transformer_block import (
    TransformerBlock,
)
from finbehavior.models.user_sequence_embedding import (
    UserSequenceEmbedding,
)
from finbehavior.tensorization.types import (
    TensorizedProfile,
    TensorizedUser,
)
from finbehavior.tokenization.special_tokens import (
    USR_TOKEN,
)
from finbehavior.tokenization.vocabulary import (
    build_vocabulary,
)


def build_transformer_block():
    self_attention = SelfAttention()

    self_attention_block = SelfAttentionBlock(
        self_attention=self_attention,
    )

    feed_forward = FeedForward()

    return TransformerBlock(
        self_attention_block=self_attention_block,
        feed_forward=feed_forward,
    )


def test_finbehavior_model_encodes_user():
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
            build_transformer_block(),
            build_transformer_block(),
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
