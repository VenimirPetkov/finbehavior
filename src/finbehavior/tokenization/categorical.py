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
from finbehavior.tokenization.profile_values import get_balance_quantile_tokens


def get_categorical_tokens() -> tuple[str, ...]:
    trading_instruments = tuple(
        instrument
        for instruments in TRADING_INSTRUMENTS.values()
        for instrument in instruments
    )

    trading_asset_classes = tuple(TRADING_INSTRUMENTS.keys())

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
        *trading_asset_classes,
        *trading_instruments,
        *TRADING_CURRENCIES,
        *COMMUNICATION_CHANNELS,
        *COMMUNICATION_TOPICS,
        *COMMUNICATION_ENGAGEMENTS,
        *PLAN_VALUES,
        *get_balance_quantile_tokens(),
    )
