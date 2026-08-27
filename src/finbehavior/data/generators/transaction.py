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
from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event


def _choose_foreign_region(
    home_region: str,
    rng: random.Random,
) -> str:
    foreign_regions = tuple(region for region in REGIONS if region != home_region)

    return rng.choice(foreign_regions)


def _choose_foreign_currency(
    home_currency: str,
    rng: random.Random,
) -> str:
    foreign_currencies = tuple(
        currency for currency in SUPPORTED_CURRENCIES if currency != home_currency
    )

    return rng.choice(foreign_currencies)


def generate_card_payment(
    behavior: BehaviorProfile,
    created: datetime,
    home_region: str,
    rng: random.Random,
) -> Event:
    home_currency = BASE_CURRENCY_BY_REGION[home_region]

    foreign_payment_probability = behavior.travel_tendency * 0.35

    is_foreign = rng.random() < foreign_payment_probability

    if is_foreign:
        merchant_region = _choose_foreign_region(
            home_region,
            rng,
        )
        currency = BASE_CURRENCY_BY_REGION[merchant_region]
    else:
        merchant_region = home_region
        currency = home_currency

    travel_category_probability = behavior.travel_tendency * 0.40

    if rng.random() < travel_category_probability:
        merchant_category = rng.choice(TRAVEL_MERCHANT_CATEGORIES)
    else:
        merchant_category = rng.choice(COMMON_MERCHANT_CATEGORIES)

    amount = round(
        rng.uniform(5.0, 120.0) * (0.5 + behavior.spending_tendency),
        2,
    )

    return Event(
        created=created,
        source=EventSource.TRANSACTION,
        fields={
            "type": "card_payment",
            "direction": "out",
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

    direction = rng.choice(("in", "out"))

    amount = round(
        rng.uniform(20.0, 800.0) * (0.5 + behavior.income_level),
        2,
    )

    return Event(
        created=created,
        source=EventSource.TRANSACTION,
        fields={
            "type": "transfer",
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
        rng.uniform(20.0, 300.0) * (0.5 + behavior.spending_tendency),
        2,
    )

    return Event(
        created=created,
        source=EventSource.TRANSACTION,
        fields={
            "type": "cash_withdrawal",
            "direction": "out",
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
    from_currency = BASE_CURRENCY_BY_REGION[home_region]

    to_currency = _choose_foreign_currency(
        from_currency,
        rng,
    )

    amount = round(
        rng.uniform(50.0, 1500.0) * (0.5 + behavior.income_level),
        2,
    )

    return Event(
        created=created,
        source=EventSource.TRANSACTION,
        fields={
            "type": "fx_exchange",
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
        population=(
            "card_payment",
            "transfer",
            "cash_withdrawal",
            "fx_exchange",
        ),
        weights=(
            5.0 + behavior.spending_tendency * 3.0,
            2.0,
            1.0,
            0.5 + behavior.travel_tendency * 3.0,
        ),
        k=1,
    )[0]

    if transaction_type == "card_payment":
        return generate_card_payment(
            behavior,
            created,
            home_region,
            rng,
        )

    if transaction_type == "transfer":
        return generate_transfer(
            behavior,
            created,
            home_region,
            rng,
        )

    if transaction_type == "cash_withdrawal":
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
