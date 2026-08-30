from datetime import datetime

import torch

from finbehavior.data.generators.dataset import (
    generate_dataset,
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
from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
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
from finbehavior.tensorization.user import (
    tensorize_user,
)
from finbehavior.tokenization.fit import (
    fit_numerical_tokenization,
)
from finbehavior.tokenization.numerical import (
    QuantileBucketizer,
)
from finbehavior.tokenization.special_tokens import (
    MASK_TOKEN,
)
from finbehavior.tokenization.user import (
    tokenize_user_record,
)
from finbehavior.tokenization.vocabulary import (
    build_vocabulary,
)
from finbehavior.training.masked_value_training_loop import (
    train_masked_values,
)
from finbehavior.training.masking import (
    mask_event_value,
)


def build_transformer_block():
    return TransformerBlock(
        self_attention_block=SelfAttentionBlock(
            self_attention=SelfAttention(),
        ),
        feed_forward=FeedForward(),
    )


def build_model(
    vocabulary_size: int,
):
    field_embedding = FieldEmbedding(
        vocabulary_size=vocabulary_size,
    )

    profile_embedding = ProfileEmbedding(
        field_embedding=field_embedding,
    )

    event_embedding = EventEmbedding(
        field_embedding=field_embedding,
        temporal_projection=TemporalProjection(),
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

    return model, field_embedding


def test_real_finbehavior_model_can_train_end_to_end():
    torch.manual_seed(0)

    synthetic_users = generate_dataset(
        number_of_users=20,
        start=datetime(2026, 1, 1),
        evaluation_point=datetime(2026, 1, 15),
        seed=42,
    )

    records = tuple(synthetic_user.record for synthetic_user in synthetic_users)

    vocabulary = build_vocabulary()

    bucketizer = QuantileBucketizer(
        number_of_buckets=4,
    )

    fit_numerical_tokenization(
        records=records,
        bucketizer=bucketizer,
        vocabulary=vocabulary,
    )

    tensorized_users = tuple(
        tensorize_user(
            tokenize_user_record(
                record=record,
                vocabulary=vocabulary,
                numerical_bucketizer=bucketizer,
            )
        )
        for record in records
    )

    training_users = tuple(
        user
        for user in tensorized_users
        if any(event.value_ids.numel() > 0 for event in user.events)
    )

    training_user = min(
        training_users,
        key=lambda user: len(user.events),
    )

    event_index = next(
        index
        for index, event in enumerate(training_user.events)
        if event.value_ids.numel() > 0
    )

    example = mask_event_value(
        user=training_user,
        event_index=event_index,
        field_index=0,
        mask_token_id=vocabulary.get_id(MASK_TOKEN),
    )

    model, field_embedding = build_model(
        vocabulary_size=len(vocabulary),
    )

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=len(vocabulary),
    )

    embedding_before = field_embedding.token_embedding.embedding.weight.detach().clone()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(prediction_head.parameters()),
        lr=0.01,
    )

    losses = train_masked_values(
        model=model,
        prediction_head=prediction_head,
        optimizer=optimizer,
        examples=(example,),
        epoch_count=20,
    )

    assert losses[-1] < losses[0]

    assert not torch.allclose(
        field_embedding.token_embedding.embedding.weight,
        embedding_before,
    )
