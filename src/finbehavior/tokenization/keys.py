from finbehavior.data.reference.field_keys import (
    EVENT_FIELD_KEYS_BY_SOURCE,
    PROFILE_FIELD_KEYS,
)
from finbehavior.domain.enums import EventSource


def get_event_key_token(source: EventSource, field: str) -> str:
    return f"{source.value}.{field}"


def get_key_tokens() -> tuple[str, ...]:
    event_key_tokens = tuple(
        get_event_key_token(source, field)
        for source, fields in EVENT_FIELD_KEYS_BY_SOURCE.items()
        for field in fields
    )

    return (*event_key_tokens, *PROFILE_FIELD_KEYS)
