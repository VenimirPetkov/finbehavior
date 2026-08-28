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


@dataclass(frozen=True)
class TokenizedProfile:
    user_token_id: int
    fields: tuple[TokenizedField, ...]


@dataclass(frozen=True)
class TokenizedUser:
    user_id: int
    profile: TokenizedProfile
    events: tuple[TokenizedEvent, ...]
