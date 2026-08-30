import torch

from .types import (
    TensorizedEvent,
    TensorizedProfile,
    TensorizedUser,
)


def move_profile_to_device(
    profile: TensorizedProfile,
    device: torch.device,
) -> TensorizedProfile:
    return TensorizedProfile(
        user_token_id=profile.user_token_id.to(device),
        key_ids=profile.key_ids.to(device),
        value_ids=profile.value_ids.to(device),
    )


def move_event_to_device(
    event: TensorizedEvent,
    device: torch.device,
) -> TensorizedEvent:
    return TensorizedEvent(
        event_token_id=event.event_token_id.to(device),
        key_ids=event.key_ids.to(device),
        value_ids=event.value_ids.to(device),
        calendar_features=event.calendar_features.to(device),
        elapsed_time_feature=event.elapsed_time_feature.to(device),
    )


def move_user_to_device(
    user: TensorizedUser,
    device: torch.device,
) -> TensorizedUser:
    return TensorizedUser(
        user_id=user.user_id,
        profile=move_profile_to_device(
            profile=user.profile,
            device=device,
        ),
        events=tuple(
            move_event_to_device(
                event=event,
                device=device,
            )
            for event in user.events
        ),
    )
