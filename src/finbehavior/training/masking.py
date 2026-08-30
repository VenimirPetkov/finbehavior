from dataclasses import dataclass

import torch

from finbehavior.tensorization.types import (
    TensorizedEvent,
    TensorizedUser,
)


@dataclass(frozen=True)
class MaskedValueExample:
    user: TensorizedUser
    target_token_id: torch.Tensor
    event_index: int
    field_index: int


def mask_event_value(
    user: TensorizedUser,
    event_index: int,
    field_index: int,
    mask_token_id: int,
) -> MaskedValueExample:
    if event_index < 0 or event_index >= len(user.events):
        raise IndexError("Event index out of range")

    event = user.events[event_index]

    if field_index < 0 or field_index >= event.value_ids.numel():
        raise IndexError("Field index out of range")

    target_token_id = event.value_ids[field_index].clone()

    masked_value_ids = event.value_ids.clone()

    masked_value_ids[field_index] = mask_token_id

    masked_event = TensorizedEvent(
        event_token_id=event.event_token_id,
        key_ids=event.key_ids,
        value_ids=masked_value_ids,
        calendar_features=event.calendar_features,
        elapsed_time_feature=(event.elapsed_time_feature),
    )

    masked_events = list(user.events)
    masked_events[event_index] = masked_event

    masked_user = TensorizedUser(
        user_id=user.user_id,
        profile=user.profile,
        events=tuple(masked_events),
    )

    return MaskedValueExample(
        user=masked_user,
        target_token_id=target_token_id,
        event_index=event_index,
        field_index=field_index,
    )
