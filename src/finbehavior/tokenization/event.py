from datetime import datetime

from finbehavior.data.reference.field_keys import (
    EVENT_FIELD_KEYS_BY_SOURCE,
)
from finbehavior.domain.event import Event

from .keys import get_event_key_token
from .numerical import QuantileBucketizer
from .special_tokens import EVT_TOKEN
from .temporal import (
    get_calendar_features,
    seconds_to_latest_event,
    soft_log_elapsed_seconds,
)
from .types import TokenizedEvent, TokenizedField
from .vocabulary import Vocabulary


def tokenize_event(
    event: Event,
    latest_event_time: datetime,
    vocabulary: Vocabulary,
    numerical_bucketizer: QuantileBucketizer,
) -> TokenizedEvent:
    allowed_fields = EVENT_FIELD_KEYS_BY_SOURCE[event.source]

    unknown_fields = set(event.fields) - set(allowed_fields)

    if unknown_fields:
        raise ValueError(
            f"Unknown fields for {event.source.value}: " f"{sorted(unknown_fields)}"
        )

    tokenized_fields = tuple(
        _tokenize_field(
            event=event,
            field_name=field_name,
            vocabulary=vocabulary,
            numerical_bucketizer=numerical_bucketizer,
        )
        for field_name in allowed_fields
        if field_name in event.fields
    )

    elapsed_seconds = seconds_to_latest_event(
        event_time=event.created,
        latest_event_time=latest_event_time,
    )

    return TokenizedEvent(
        event_token_id=vocabulary.get_id(EVT_TOKEN),
        fields=tokenized_fields,
        calendar_features=get_calendar_features(event.created),
        elapsed_time_feature=soft_log_elapsed_seconds(elapsed_seconds),
    )


def _tokenize_field(
    event: Event,
    field_name: str,
    vocabulary: Vocabulary,
    numerical_bucketizer: QuantileBucketizer,
) -> TokenizedField:
    key_token = get_event_key_token(
        event.source,
        field_name,
    )

    key_id = vocabulary.get_id(key_token)

    value = event.fields[field_name]

    value_id = _encode_value(
        key_token=key_token,
        value=value,
        vocabulary=vocabulary,
        numerical_bucketizer=numerical_bucketizer,
    )

    return TokenizedField(
        key_id=key_id,
        value_id=value_id,
    )


def _encode_value(
    key_token: str,
    value: str | int | float | bool,
    vocabulary: Vocabulary,
    numerical_bucketizer: QuantileBucketizer,
) -> int:
    if isinstance(value, bool):
        raise TypeError("Boolean event values are not supported yet")

    if isinstance(value, (int, float)):
        bucket_token = numerical_bucketizer.transform(
            key_token,
            value,
        )

        return vocabulary.get_id(bucket_token)

    if isinstance(value, str):
        return vocabulary.encode(value)

    raise TypeError("Unsupported event value type: " f"{type(value).__name__}")
