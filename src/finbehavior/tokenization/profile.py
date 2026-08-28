from finbehavior.data.reference.field_keys import (
    BALANCE_QUANTILE_FIELD,
    PROFILE_FIELD_KEYS,
)
from finbehavior.domain.profile import ProfileState

from .profile_values import (
    get_balance_quantile_token,
)
from .special_tokens import USR_TOKEN
from .types import (
    TokenizedField,
    TokenizedProfile,
)
from .vocabulary import Vocabulary


def tokenize_profile(
    profile: ProfileState,
    vocabulary: Vocabulary,
) -> TokenizedProfile:
    unknown_fields = set(profile.fields) - set(PROFILE_FIELD_KEYS)

    if unknown_fields:
        raise ValueError("Unknown profile fields: " f"{sorted(unknown_fields)}")

    tokenized_fields = tuple(
        _tokenize_profile_field(
            field_name=field_name,
            value=profile.fields[field_name],
            vocabulary=vocabulary,
        )
        for field_name in PROFILE_FIELD_KEYS
        if field_name in profile.fields
    )

    return TokenizedProfile(
        user_token_id=vocabulary.get_id(USR_TOKEN),
        fields=tokenized_fields,
    )


def _tokenize_profile_field(
    field_name: str,
    value: str | int | float | bool,
    vocabulary: Vocabulary,
) -> TokenizedField:
    key_id = vocabulary.get_id(field_name)

    if field_name == BALANCE_QUANTILE_FIELD:
        value_token = get_balance_quantile_token(value)

        value_id = vocabulary.get_id(value_token)

    else:
        if not isinstance(value, str):
            raise TypeError(
                f"Profile field '{field_name}' " "must contain a string value"
            )

        value_id = vocabulary.encode(value)

    return TokenizedField(
        key_id=key_id,
        value_id=value_id,
    )
