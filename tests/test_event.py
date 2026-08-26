from datetime import datetime

from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event


def test_create_transaction_event():
    created = datetime(2026, 8, 26, 14, 30)

    event = Event(
        created=created,
        source=EventSource.TRANSACTION,
        fields={
            "type": "card_payment",
            "currency": "EUR",
            "amount": 42.50,
            "direction": "out",
        },
    )

    assert event.created == created
    assert event.source == EventSource.TRANSACTION
    assert event.fields["type"] == "card_payment"
    assert event.fields["currency"] == "EUR"
    assert event.fields["amount"] == 42.50
    assert event.fields["direction"] == "out"

def test_create_trading_event_with_different_fields():
    event = Event(
        created=datetime(2026, 8, 26, 15, 0),
        source=EventSource.TRADING,
        fields={
            "type": "buy",
            "symbol": "AAPL",
            "quantity": 5,
            "price": 225.50,
        },
    )

    assert event.source == EventSource.TRADING
    assert event.fields["type"] == "buy"
    assert event.fields["symbol"] == "AAPL"
    assert event.fields["quantity"] == 5
    assert event.fields["price"] == 225.50