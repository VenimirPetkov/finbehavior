import torch

from finbehavior.data.reference.field_keys import (
    DIRECTION_FIELD,
    TYPE_FIELD,
)
from finbehavior.data.reference.transaction import (
    TRANSACTION_DIRECTION_OUT,
    TRANSACTION_TYPE_CARD_PAYMENT,
)
from finbehavior.domain.enums import EventSource
from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.models.event_embedding import (
    EventEmbedding,
)
from finbehavior.models.field_embedding import (
    FieldEmbedding,
)
from finbehavior.models.temporal_projection import (
    TemporalProjection,
)
from finbehavior.tensorization.types import (
    TensorizedEvent,
)
from finbehavior.tokenization.keys import (
    get_event_key_token,
)
from finbehavior.tokenization.special_tokens import (
    EVT_TOKEN,
)
from finbehavior.tokenization.vocabulary import (
    build_vocabulary,
)


def test_event_embedding():
    vocabulary = build_vocabulary()

    field_embedding = FieldEmbedding(
        vocabulary_size=len(vocabulary),
    )

    temporal_projection = TemporalProjection()

    event_embedding = EventEmbedding(
        field_embedding=field_embedding,
        temporal_projection=temporal_projection,
    )

    type_key_id = vocabulary.get_id(
        get_event_key_token(
            EventSource.TRANSACTION,
            TYPE_FIELD,
        )
    )

    direction_key_id = vocabulary.get_id(
        get_event_key_token(
            EventSource.TRANSACTION,
            DIRECTION_FIELD,
        )
    )

    card_payment_id = vocabulary.get_id(TRANSACTION_TYPE_CARD_PAYMENT)

    direction_out_id = vocabulary.get_id(TRANSACTION_DIRECTION_OUT)

    event = TensorizedEvent(
        event_token_id=torch.tensor(
            vocabulary.get_id(EVT_TOKEN),
            dtype=torch.long,
        ),
        key_ids=torch.tensor(
            [
                type_key_id,
                direction_key_id,
            ],
            dtype=torch.long,
        ),
        value_ids=torch.tensor(
            [
                card_payment_id,
                direction_out_id,
            ],
            dtype=torch.long,
        ),
        calendar_features=torch.tensor(
            [
                0.5,
                -0.5,
                0.25,
                -0.25,
                1.0,
                0.0,
            ],
            dtype=torch.float32,
        ),
        elapsed_time_feature=torch.tensor(
            17.42,
            dtype=torch.float32,
        ),
    )

    event_vector = event_embedding(event)

    assert event_vector.shape == (DEFAULT_EMBEDDING_DIMENSION,)

    assert event_embedding.composition.weight.requires_grad is True
