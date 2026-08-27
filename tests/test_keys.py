from finbehavior.data.reference.field_keys import (
    AMOUNT_FIELD,
    CURRENCY_FIELD,
    TYPE_FIELD,
)
from finbehavior.tokenization.keys import get_key_tokens


def test_get_key_tokens_contains_expected_fields():
    keys = get_key_tokens()

    assert TYPE_FIELD in keys
    assert CURRENCY_FIELD in keys
    assert AMOUNT_FIELD in keys