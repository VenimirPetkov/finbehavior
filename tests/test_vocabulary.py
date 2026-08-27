from finbehavior.tokenization.special_tokens import (
    EVT_TOKEN,
    MASK_TOKEN,
    UNK_TOKEN,
    USR_TOKEN,
)
from finbehavior.tokenization.vocabulary import Vocabulary


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