from .config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from .config.encoder import (
    DEFAULT_TRANSFORMER_BLOCK_COUNT,
)
from .encoder import TransformerEncoder
from .event_embedding import EventEmbedding
from .feed_forward import FeedForward
from .field_embedding import FieldEmbedding
from .history_embedding import HistoryEmbedding
from .model import FinBehaviorModel
from .profile_embedding import ProfileEmbedding
from .self_attention import SelfAttention
from .self_attention_block import SelfAttentionBlock
from .temporal_projection import TemporalProjection
from .transformer_block import TransformerBlock
from .user_sequence_embedding import UserSequenceEmbedding


def _build_transformer_block(
    embedding_dimension: int,
) -> TransformerBlock:
    self_attention = SelfAttention(
        embedding_dimension=embedding_dimension,
    )

    self_attention_block = SelfAttentionBlock(
        self_attention=self_attention,
        embedding_dimension=embedding_dimension,
    )

    feed_forward = FeedForward(
        embedding_dimension=embedding_dimension,
    )

    return TransformerBlock(
        self_attention_block=self_attention_block,
        feed_forward=feed_forward,
        embedding_dimension=embedding_dimension,
    )


def build_finbehavior_model(
    vocabulary_size: int,
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    transformer_block_count: int = DEFAULT_TRANSFORMER_BLOCK_COUNT,
) -> FinBehaviorModel:
    if vocabulary_size <= 0:
        raise ValueError("Vocabulary size must be positive")

    if transformer_block_count <= 0:
        raise ValueError("Transformer block count must be positive")

    field_embedding = FieldEmbedding(
        vocabulary_size=vocabulary_size,
        embedding_dimension=embedding_dimension,
    )

    profile_embedding = ProfileEmbedding(
        field_embedding=field_embedding,
    )

    temporal_projection = TemporalProjection(
        embedding_dimension=embedding_dimension,
    )

    event_embedding = EventEmbedding(
        field_embedding=field_embedding,
        temporal_projection=temporal_projection,
        embedding_dimension=embedding_dimension,
    )

    history_embedding = HistoryEmbedding(
        event_embedding=event_embedding,
    )

    user_sequence_embedding = UserSequenceEmbedding(
        profile_embedding=profile_embedding,
        history_embedding=history_embedding,
    )

    blocks = tuple(
        _build_transformer_block(
            embedding_dimension=embedding_dimension,
        )
        for _ in range(transformer_block_count)
    )

    encoder = TransformerEncoder(
        blocks=blocks,
    )

    return FinBehaviorModel(
        user_sequence_embedding=user_sequence_embedding,
        encoder=encoder,
    )
