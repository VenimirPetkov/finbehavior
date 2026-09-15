from typing import TYPE_CHECKING

from finbehavior.data.reference.app import (
    APP_ACTIONS,
    APP_SCREENS,
)
from finbehavior.data.reference.communication import (
    COMMUNICATION_CHANNELS,
    COMMUNICATION_ENGAGEMENTS,
    COMMUNICATION_TOPICS,
)
from finbehavior.data.reference.currencies import (
    SUPPORTED_CURRENCIES,
)
from finbehavior.data.reference.merchant_categories import (
    COMMON_MERCHANT_CATEGORIES,
    TRAVEL_MERCHANT_CATEGORIES,
)
from finbehavior.data.reference.profile import PLAN_VALUES
from finbehavior.data.reference.regions import REGIONS
from finbehavior.data.reference.trading import (
    TRADING_ACTIONS,
    TRADING_CURRENCIES,
    TRADING_INSTRUMENTS,
)
from finbehavior.data.reference.transaction import (
    TRANSACTION_DIRECTIONS,
    TRANSACTION_TYPES,
)
from finbehavior.domain.enums import EventSource
from finbehavior.tokenization.keys import get_event_key_token
from finbehavior.tokenization.profile_values import get_balance_quantile_tokens

if TYPE_CHECKING:
    from finbehavior.tokenization.vocabulary import Vocabulary


def _flatten_trading_instruments() -> tuple[str, ...]:
    return tuple(
        instrument
        for instruments in TRADING_INSTRUMENTS.values()
        for instrument in instruments
    )


_TRANSACTION_CATEGORIES = (*COMMON_MERCHANT_CATEGORIES, *TRAVEL_MERCHANT_CATEGORIES)

_CATEGORICAL_VALUE_TOKENS_BY_KEY: dict[str, tuple[str, ...]] = {
    get_event_key_token(EventSource.TRANSACTION, "type"): TRANSACTION_TYPES,
    get_event_key_token(EventSource.TRANSACTION, "direction"): (TRANSACTION_DIRECTIONS),
    get_event_key_token(EventSource.TRANSACTION, "currency"): SUPPORTED_CURRENCIES,
    get_event_key_token(EventSource.TRANSACTION, "merchant_category"): (
        _TRANSACTION_CATEGORIES
    ),
    get_event_key_token(EventSource.TRANSACTION, "merchant_region"): REGIONS,
    get_event_key_token(EventSource.TRANSACTION, "atm_region"): REGIONS,
    get_event_key_token(EventSource.TRANSACTION, "from_currency"): (
        SUPPORTED_CURRENCIES
    ),
    get_event_key_token(EventSource.TRANSACTION, "to_currency"): (SUPPORTED_CURRENCIES),
    get_event_key_token(EventSource.APP, "screen"): APP_SCREENS,
    get_event_key_token(EventSource.APP, "action"): APP_ACTIONS,
    get_event_key_token(EventSource.TRADING, "action"): TRADING_ACTIONS,
    get_event_key_token(EventSource.TRADING, "asset_class"): tuple(TRADING_INSTRUMENTS),
    get_event_key_token(EventSource.TRADING, "instrument"): (
        _flatten_trading_instruments()
    ),
    get_event_key_token(EventSource.TRADING, "currency"): TRADING_CURRENCIES,
    get_event_key_token(EventSource.COMMUNICATION, "channel"): (COMMUNICATION_CHANNELS),
    get_event_key_token(EventSource.COMMUNICATION, "topic"): COMMUNICATION_TOPICS,
    get_event_key_token(EventSource.COMMUNICATION, "engagement"): (
        COMMUNICATION_ENGAGEMENTS
    ),
    "plan": PLAN_VALUES,
    "region": REGIONS,
    "balance_quantile": get_balance_quantile_tokens(),
}


def get_categorical_value_tokens_by_key() -> dict[str, tuple[str, ...]]:
    """Return the allowed categorical value tokens for every categorical key."""

    return dict(_CATEGORICAL_VALUE_TOKENS_BY_KEY)


def get_categorical_value_tokens(key_token: str) -> tuple[str, ...]:
    """Return the categorical domain for ``key_token``.

    ``KeyError`` deliberately distinguishes numerical or unknown keys from
    categorical keys. Callers constructing field-specific output masks can
    therefore handle numerical bucket domains separately.
    """

    return _CATEGORICAL_VALUE_TOKENS_BY_KEY[key_token]


def get_categorical_value_ids(
    key_token: str,
    vocabulary: "Vocabulary",
) -> tuple[int, ...]:
    """Return vocabulary ids allowed for a categorical field."""

    return tuple(
        vocabulary.get_id(token) for token in get_categorical_value_tokens(key_token)
    )


def get_categorical_tokens() -> tuple[str, ...]:
    # This order is part of persisted tokenizer/checkpoint compatibility.
    return (
        *TRANSACTION_TYPES,
        *TRANSACTION_DIRECTIONS,
        *SUPPORTED_CURRENCIES,
        *REGIONS,
        *COMMON_MERCHANT_CATEGORIES,
        *TRAVEL_MERCHANT_CATEGORIES,
        *APP_SCREENS,
        *APP_ACTIONS,
        *TRADING_ACTIONS,
        *tuple(TRADING_INSTRUMENTS),
        *_flatten_trading_instruments(),
        *TRADING_CURRENCIES,
        *COMMUNICATION_CHANNELS,
        *COMMUNICATION_TOPICS,
        *COMMUNICATION_ENGAGEMENTS,
        *PLAN_VALUES,
        *get_balance_quantile_tokens(),
    )
