import pytest

from finbehavior.domain.profile import ProfileState
from finbehavior.tokenization.profile import (
    tokenize_profile,
)
from finbehavior.tokenization.special_tokens import (
    USR_TOKEN,
)
from finbehavior.tokenization.vocabulary import (
    build_vocabulary,
)


def test_tokenize_profile():
    profile = ProfileState(
        fields={
            "plan": "premium",
            "region": "ES",
            "balance_quantile": 7,
        }
    )

    vocabulary = build_vocabulary()

    tokenized = tokenize_profile(
        profile=profile,
        vocabulary=vocabulary,
    )

    assert tokenized.user_token_id == (vocabulary.get_id(USR_TOKEN))

    assert len(tokenized.fields) == 3

    decoded_fields = {
        vocabulary.get_token(field.key_id): vocabulary.get_token(field.value_id)
        for field in tokenized.fields
    }

    assert decoded_fields == {
        "plan": "premium",
        "region": "ES",
        "balance_quantile": "balance_quantile_7",
    }


def test_balance_quantile_tokens_are_in_vocabulary():
    vocabulary = build_vocabulary()

    token_id = vocabulary.get_id("balance_quantile_7")

    assert vocabulary.get_token(token_id) == "balance_quantile_7"


def test_rejects_invalid_balance_quantile():
    profile = ProfileState(
        fields={
            "plan": "premium",
            "region": "ES",
            "balance_quantile": 10,
        }
    )

    vocabulary = build_vocabulary()

    with pytest.raises(
        ValueError,
        match="Invalid balance quantile",
    ):
        tokenize_profile(
            profile=profile,
            vocabulary=vocabulary,
        )


def test_rejects_unknown_profile_field():
    profile = ProfileState(
        fields={
            "plan": "premium",
            "region": "ES",
            "balance_quantile": 7,
            "banana": "yes",
        }
    )

    vocabulary = build_vocabulary()

    with pytest.raises(
        ValueError,
        match="Unknown profile fields",
    ):
        tokenize_profile(
            profile=profile,
            vocabulary=vocabulary,
        )
