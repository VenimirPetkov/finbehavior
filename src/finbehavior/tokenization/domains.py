from finbehavior.data.reference.field_keys import (
    EVENT_FIELD_KEYS_BY_SOURCE,
    NUMERICAL_FIELD_KEYS_BY_SOURCE,
)

from .categorical import get_categorical_value_tokens
from .keys import get_event_key_token
from .numerical import QuantileBucketizer
from .special_tokens import UNK_TOKEN
from .vocabulary import Vocabulary


def build_event_value_id_domains(
    vocabulary: Vocabulary,
    bucketizer: QuantileBucketizer,
) -> dict[int, tuple[int, ...]]:
    """Map every event key id to its valid prediction value ids.

    Categorical domains include ``[UNK]`` because tokenization deliberately
    maps out-of-domain input values to that safe fallback. Numerical domains
    contain only the fitted bucket ids.
    """

    domains: dict[int, tuple[int, ...]] = {}
    unknown_id = vocabulary.get_id(UNK_TOKEN)

    for source, fields in EVENT_FIELD_KEYS_BY_SOURCE.items():
        numerical_fields = NUMERICAL_FIELD_KEYS_BY_SOURCE.get(source, ())

        for field_name in fields:
            key_token = get_event_key_token(source, field_name)
            key_id = vocabulary.get_id(key_token)

            if field_name in numerical_fields:
                value_tokens = bucketizer.get_bucket_tokens(key_token)
                value_ids = tuple(vocabulary.get_id(token) for token in value_tokens)
            else:
                categorical_tokens = get_categorical_value_tokens(key_token)
                value_ids = (
                    *(vocabulary.get_id(token) for token in categorical_tokens),
                    unknown_id,
                )

            domains[key_id] = value_ids

    return domains
