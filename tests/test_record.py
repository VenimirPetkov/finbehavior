from datetime import datetime

from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event
from finbehavior.domain.profile import ProfileState
from finbehavior.domain.record import UserRecord


def test_create_user_record():
    profile = ProfileState(
        fields={
            "plan": "premium",
            "region": "ES",
        }
    )

    event = Event(
        created=datetime(2026, 8, 25, 10, 30),
        source=EventSource.TRANSACTION,
        fields={
            "type": "card_payment",
            "amount": 42.50,
            "currency": "EUR",
        },
    )

    record = UserRecord(
        user_id=42,
        evaluation_point=datetime(2026, 8, 26, 12, 0),
        profile=profile,
        events=[event],
    )

    assert record.user_id == 42
    assert record.profile == profile
    assert record.events == [event]
    assert record.evaluation_point == datetime(2026, 8, 26, 12, 0)