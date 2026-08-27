import random
from datetime import datetime

from finbehavior.data.behavior_profile import BehaviorProfile
from finbehavior.data.reference.currencies import (
    BASE_CURRENCY_BY_REGION,
    SUPPORTED_CURRENCIES,
)
from finbehavior.data.reference.merchant_categories import (
    COMMON_MERCHANT_CATEGORIES,
    TRAVEL_MERCHANT_CATEGORIES,
)
from finbehavior.data.reference.regions import REGIONS
from finbehavior.data.reference.transaction import (
    TRANSACTION_DIRECTION_OUT,
    TRANSACTION_DIRECTIONS,
    TRANSACTION_TYPE_CARD_PAYMENT,
    TRANSACTION_TYPE_CASH_WITHDRAWAL,
    TRANSACTION_TYPE_FX_EXCHANGE,
    TRANSACTION_TYPE_TRANSFER,
    TRANSACTION_TYPES,
)
from finbehavior.data.reference.transaction_generation import (
    AMOUNT_DECIMAL_PLACES,
    BEHAVIOR_SCALE_BASE,
    CARD_PAYMENT_BASE_WEIGHT,
    CARD_PAYMENT_MAX_AMOUNT,
    CARD_PAYMENT_MIN_AMOUNT,
    CARD_PAYMENT_SPENDING_WEIGHT_MULTIPLIER,
    CASH_WITHDRAWAL_MAX_AMOUNT,
    CASH_WITHDRAWAL_MIN_AMOUNT,
    CASH_WITHDRAWAL_WEIGHT,
    FOREIGN_PAYMENT_TRAVEL_PROBABILITY_MULTIPLIER,
    FX_EXCHANGE_BASE_WEIGHT,
    FX_EXCHANGE_MAX_AMOUNT,
    FX_EXCHANGE_MIN_AMOUNT,
    FX_EXCHANGE_TRAVEL_WEIGHT_MULTIPLIER,
    TRANSFER_MAX_AMOUNT,
    TRANSFER_MIN_AMOUNT,
    TRANSFER_WEIGHT,
    TRAVEL_CATEGORY_PROBABILITY_MULTIPLIER,
)
from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event


def _choose_foreign_region(
    home_region: str,
    rng: random.Random,
) -> str:
    foreign_regions = tuple(
        region
        for region in REGIONS
        if region != home_region
    )

    return rng.choice(foreign_regions)


def _choose_foreign_currency(
    home_currency: str,
    rng: random.Random,
) -> str:
    foreign_currencies = tuple(
        currency
        for currency in SUPPORTED_CURRENCIES
        if currency != home_currency
    )

    return rng.choice(foreign_currencies)


def generate_card_payment(
    behavior: BehaviorProfile,
    created: datetime,
    home_region: str,
    rng: random.Random,
) -> Event:
    home_currency = BASE_CURRENCY_BY_REGION[home_region]

    foreign_payment_probability = (
        behavior.travel_tendency
        * FOREIGN_PAYMENT_TRAVEL_PROBABILITY_MULTIPLIER
    )

    is_foreign = (
        rng.random()
        < foreign_payment_probability
    )

    if is_foreign:
        merchant_region = _choose_foreign_region(
            home_region,
            rng,
        )

        currency = BASE_CURRENCY_BY_REGION[
            merchant_region
        ]
    else:
        merchant_region = home_region
        currency = home_currency

    travel_category_probability = (
        behavior.travel_tendency
        * TRAVEL_CATEGORY_PROBABILITY_MULTIPLIER
    )

    if rng.random() < travel_category_probability:
        merchant_category = rng.choice(
            TRAVEL_MERCHANT_CATEGORIES
        )
    else:
        merchant_category = rng.choice(
            COMMON_MERCHANT_CATEGORIES
        )

    amount = round(
        rng.uniform(
            CARD_PAYMENT_MIN_AMOUNT,
            CARD_PAYMENT_MAX_AMOUNT,
        )
        * (
            BEHAVIOR_SCALE_BASE
            + behavior.spending_tendency
        ),
        AMOUNT_DECIMAL_PLACES,
    )

    return Event(
        created=created,
        source=EventSource.TRANSACTION,
        fields={
            "type": TRANSACTION_TYPE_CARD_PAYMENT,
            "direction": TRANSACTION_DIRECTION_OUT,
            "amount": amount,
            "currency": currency,
            "merchant_category": merchant_category,
            "merchant_region": merchant_region,
        },
    )


def generate_transfer(
    behavior: BehaviorProfile,
    created: datetime,
    home_region: str,
    rng: random.Random,
) -> Event:
    currency = BASE_CURRENCY_BY_REGION[home_region]

    direction = rng.choice(
        TRANSACTION_DIRECTIONS
    )

    amount = round(
        rng.uniform(
            TRANSFER_MIN_AMOUNT,
            TRANSFER_MAX_AMOUNT,
        )
        * (
            BEHAVIOR_SCALE_BASE
            + behavior.income_level
        ),
        AMOUNT_DECIMAL_PLACES,
    )

    return Event(
        created=created,
        source=EventSource.TRANSACTION,
        fields={
            "type": TRANSACTION_TYPE_TRANSFER,
            "direction": direction,
            "amount": amount,
            "currency": currency,
        },
    )


def generate_cash_withdrawal(
    behavior: BehaviorProfile,
    created: datetime,
    home_region: str,
    rng: random.Random,
) -> Event:
    currency = BASE_CURRENCY_BY_REGION[home_region]

    amount = round(
        rng.uniform(
            CASH_WITHDRAWAL_MIN_AMOUNT,
            CASH_WITHDRAWAL_MAX_AMOUNT,
        )
        * (
            BEHAVIOR_SCALE_BASE
            + behavior.spending_tendency
        ),
        AMOUNT_DECIMAL_PLACES,
    )

    return Event(
        created=created,
        source=EventSource.TRANSACTION,
        fields={
            "type": TRANSACTION_TYPE_CASH_WITHDRAWAL,
            "direction": TRANSACTION_DIRECTION_OUT,
            "amount": amount,
            "currency": currency,
            "atm_region": home_region,
        },
    )


def generate_fx_exchange(
    behavior: BehaviorProfile,
    created: datetime,
    home_region: str,
    rng: random.Random,
) -> Event:
    from_currency = BASE_CURRENCY_BY_REGION[
        home_region
    ]

    to_currency = _choose_foreign_currency(
        from_currency,
        rng,
    )

    amount = round(
        rng.uniform(
            FX_EXCHANGE_MIN_AMOUNT,
            FX_EXCHANGE_MAX_AMOUNT,
        )
        * (
            BEHAVIOR_SCALE_BASE
            + behavior.income_level
        ),
        AMOUNT_DECIMAL_PLACES,
    )

    return Event(
        created=created,
        source=EventSource.TRANSACTION,
        fields={
            "type": TRANSACTION_TYPE_FX_EXCHANGE,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "amount": amount,
        },
    )


def generate_transaction_event(
    behavior: BehaviorProfile,
    created: datetime,
    home_region: str,
    rng: random.Random | None = None,
) -> Event:
    if rng is None:
        rng = random.Random()

    transaction_type = rng.choices(
        population=TRANSACTION_TYPES,
        weights=(
            (
                CARD_PAYMENT_BASE_WEIGHT
                + behavior.spending_tendency
                * CARD_PAYMENT_SPENDING_WEIGHT_MULTIPLIER
            ),
            TRANSFER_WEIGHT,
            CASH_WITHDRAWAL_WEIGHT,
            (
                FX_EXCHANGE_BASE_WEIGHT
                + behavior.travel_tendency
                * FX_EXCHANGE_TRAVEL_WEIGHT_MULTIPLIER
            ),
        ),
        k=1,
    )[0]

    if (
        transaction_type
        == TRANSACTION_TYPE_CARD_PAYMENT
    ):
        return generate_card_payment(
            behavior,
            created,
            home_region,
            rng,
        )

    if (
        transaction_type
        == TRANSACTION_TYPE_TRANSFER
    ):
        return generate_transfer(
            behavior,
            created,
            home_region,
            rng,
        )

    if (
        transaction_type
        == TRANSACTION_TYPE_CASH_WITHDRAWAL
    ):
        return generate_cash_withdrawal(
            behavior,
            created,
            home_region,
            rng,
        )

    return generate_fx_exchange(
        behavior,
        created,
        home_region,
        rng,
    )