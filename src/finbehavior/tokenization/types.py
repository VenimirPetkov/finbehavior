from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizedField:
    key_id: int
    value_id: int


@dataclass(frozen=True)
class TokenizedEvent:
    event_token_id: int
    fields: tuple[TokenizedField, ...]
    calendar_features: tuple[float, ...]
    elapsed_time_feature: float