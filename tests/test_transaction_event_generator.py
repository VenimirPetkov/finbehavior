import random
from datetime import datetime

from finbehavior.data.behavior_profile import BehaviorProfile
from finbehavior.data.generators.transaction import (
    generate_card_payment,
    generate_transaction_event,
)
from finbehavior.data.reference.regions import REGIONS
from finbehavior.domain.enums import EventSource


def create_behavior() -> BehaviorProfile:
    return BehaviorProfile(
        income_level=0.7,
        spending_tendency=0.8,
        travel_tendency=0.6,
        investing_tendency=0.5,
        app_activity=0.7,
        communication_engagement=0.4,
    )


def test_generate_card_payment():
    created = datetime(2026, 8, 27, 10, 30)

    event = generate_card_payment(
        behavior=create_behavior(),
        created=created,
        home_region="ES",
        rng=random.Random(42),
    )

    assert event.created == created
    assert event.source == EventSource.TRANSACTION
    assert event.fields["type"] == "card_payment"
    assert event.fields["direction"] == "out"
    assert event.fields["amount"] > 0
    assert event.fields["merchant_region"] in REGIONS


def test_generate_transaction_event_is_reproducible():
    created = datetime(2026, 8, 27, 10, 30)

    first = generate_transaction_event(
        behavior=create_behavior(),
        created=created,
        home_region="ES",
        rng=random.Random(42),
    )

    second = generate_transaction_event(
        behavior=create_behavior(),
        created=created,
        home_region="ES",
        rng=random.Random(42),
    )

    assert first == second


def test_generated_transaction_type_is_supported():
    supported_types = {
        "card_payment",
        "transfer",
        "cash_withdrawal",
        "fx_exchange",
    }

    for seed in range(100):
        event = generate_transaction_event(
            behavior=create_behavior(),
            created=datetime(2026, 8, 27, 10, 30),
            home_region="ES",
            rng=random.Random(seed),
        )

        assert event.source == EventSource.TRANSACTION
        assert event.fields["type"] in supported_types
