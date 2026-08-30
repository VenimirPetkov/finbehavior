from finbehavior.tensorization.types import (
    TensorizedUser,
)

from .masking import (
    MaskedValueExample,
    mask_event_value,
)


def build_masked_value_examples(
    users: tuple[TensorizedUser, ...],
    mask_token_id: int,
) -> tuple[MaskedValueExample, ...]:
    examples = []

    for user in users:
        for event_index, event in enumerate(user.events):
            for field_index in range(event.value_ids.numel()):
                example = mask_event_value(
                    user=user,
                    event_index=event_index,
                    field_index=field_index,
                    mask_token_id=mask_token_id,
                )

                examples.append(example)

    return tuple(examples)
