from datetime import datetime

from finbehavior.data.behavior_profile import BehaviorProfile
from finbehavior.data.serialization.user import (
    synthetic_user_to_dict,
)
from finbehavior.data.synthetic_user import SyntheticUser
from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event
from finbehavior.domain.profile import ProfileState
from finbehavior.domain.record import UserRecord


def test_synthetic_user_to_dict():
    behavior = BehaviorProfile(
        income_level=0.7,
        spending_tendency=0.8,
        travel_tendency=0.5,
        investing_tendency=0.6,
        app_activity=0.9,
        communication_engagement=0.7,
    )

    profile = ProfileState(
        fields={
            "plan": "premium",
            "region": "ES",
            "balance_quantile": 7,
        }
    )

    event = Event(
        created=datetime(2026, 1, 15, 14, 30),
        source=EventSource.TRANSACTION,
        fields={
            "type": "card_payment",
            "amount": 42.50,
            "currency": "EUR",
        },
    )

    record = UserRecord(
        user_id=42,
        evaluation_point=datetime(2026, 7, 1),
        profile=profile,
        events=[event],
    )

    user = SyntheticUser(
        behavior=behavior,
        record=record,
    )

    data = synthetic_user_to_dict(user)

    assert data["behavior"] == {
        "income_level": 0.7,
        "spending_tendency": 0.8,
        "travel_tendency": 0.5,
        "investing_tendency": 0.6,
        "app_activity": 0.9,
        "communication_engagement": 0.7,
    }

    assert data["record"]["user_id"] == 42

    assert (
        data["record"]["evaluation_point"]
        == "2026-07-01T00:00:00"
    )

    assert data["record"]["profile"] == {
        "plan": "premium",
        "region": "ES",
        "balance_quantile": 7,
    }

    assert data["record"]["events"][0] == {
        "created": "2026-01-15T14:30:00",
        "source": "transaction",
        "fields": {
            "type": "card_payment",
            "amount": 42.50,
            "currency": "EUR",
        },
    }