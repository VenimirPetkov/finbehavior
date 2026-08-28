from finbehavior.data.reference.field_keys import (
    ACTION_FIELD,
    AMOUNT_FIELD,
    CURRENCY_FIELD,
    NUMERICAL_FIELD_KEYS_BY_SOURCE,
    PLAN_FIELD,
    TYPE_FIELD,
)
from finbehavior.domain.enums import EventSource
from finbehavior.tokenization.keys import get_event_key_token, get_key_tokens


def test_get_key_tokens_contains_expected_fields():
    keys = get_key_tokens()

    assert get_event_key_token(EventSource.TRANSACTION, TYPE_FIELD) in keys
    assert get_event_key_token(EventSource.TRANSACTION, CURRENCY_FIELD) in keys
    assert get_event_key_token(EventSource.TRANSACTION, AMOUNT_FIELD) in keys
    assert PLAN_FIELD in keys


def test_event_key_tokens_are_source_aware():
    keys = get_key_tokens()

    app_action = get_event_key_token(EventSource.APP, ACTION_FIELD)
    trading_action = get_event_key_token(EventSource.TRADING, ACTION_FIELD)
    transaction_amount = get_event_key_token(EventSource.TRANSACTION, AMOUNT_FIELD)
    trading_amount = get_event_key_token(EventSource.TRADING, AMOUNT_FIELD)

    assert app_action == "app.action"
    assert trading_action == "trading.action"
    assert transaction_amount == "transaction.amount"
    assert trading_amount == "trading.amount"
    assert len({app_action, trading_action, transaction_amount, trading_amount}) == 4
    assert ACTION_FIELD not in keys
    assert AMOUNT_FIELD not in keys


def test_numerical_field_keys_are_explicit_by_source():
    assert NUMERICAL_FIELD_KEYS_BY_SOURCE == {
        EventSource.TRANSACTION: (AMOUNT_FIELD,),
        EventSource.TRADING: (AMOUNT_FIELD,),
    }
