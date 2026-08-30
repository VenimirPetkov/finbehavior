import pytest

from finbehavior.data.reference.field_keys import CURRENCY_FIELD
from finbehavior.data.reference.profile import PLAN_VALUES
from finbehavior.domain.enums import EventSource
from finbehavior.tokenization.keys import get_event_key_token
from finbehavior.tokenization.special_tokens import (
    EVT_TOKEN,
    MASK_TOKEN,
    UNK_TOKEN,
    USR_TOKEN,
)
from finbehavior.tokenization.vocabulary import Vocabulary, build_vocabulary


def test_vocabulary_contains_special_tokens():
    vocab = Vocabulary()

    assert vocab.get_id(MASK_TOKEN) == 0
    assert vocab.get_id(UNK_TOKEN) == 1
    assert vocab.get_id(USR_TOKEN) == 2
    assert vocab.get_id(EVT_TOKEN) == 3


def test_add_token():
    vocab = Vocabulary()

    token_id = vocab.add("EUR")

    assert token_id == 4
    assert vocab.get_id("EUR") == 4
    assert vocab.get_token(4) == "EUR"


def test_adding_same_token_twice_reuses_id():
    vocab = Vocabulary()

    first_id = vocab.add("EUR")
    second_id = vocab.add("EUR")

    assert first_id == second_id
    assert len(vocab) == 5


def test_build_vocabulary_contains_keys_and_values():
    vocab = build_vocabulary()

    currency_key = get_event_key_token(
        EventSource.TRANSACTION,
        CURRENCY_FIELD,
    )
    currency_key_id = vocab.get_id(currency_key)
    eur_value_id = vocab.get_id("EUR")

    assert currency_key_id != eur_value_id
    assert vocab.get_token(currency_key_id) == currency_key
    assert vocab.get_token(eur_value_id) == "EUR"


def test_build_vocabulary_contains_profile_values():
    vocab = build_vocabulary()

    for plan in PLAN_VALUES:
        assert vocab.get_token(vocab.get_id(plan)) == plan


def test_encode_returns_known_token_id():
    vocab = Vocabulary()

    token_id = vocab.add("EUR")

    assert vocab.encode("EUR") == token_id


def test_encode_unknown_token_uses_unk():
    vocab = Vocabulary()

    assert vocab.encode("SOMETHING_NEW") == vocab.get_id(UNK_TOKEN)


def test_get_id_remains_strict_for_unknown_tokens():
    vocab = Vocabulary()

    with pytest.raises(KeyError):
        vocab.get_id("SOMETHING_NEW")
