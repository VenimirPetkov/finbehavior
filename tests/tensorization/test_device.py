import torch

from finbehavior.tensorization.device import (
    move_event_to_device,
    move_profile_to_device,
    move_user_to_device,
)
from finbehavior.tensorization.types import (
    TensorizedEvent,
    TensorizedProfile,
    TensorizedUser,
)


def get_test_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda",
            torch.cuda.current_device(),
        )

    return torch.device("cpu")


def build_profile() -> TensorizedProfile:
    return TensorizedProfile(
        user_token_id=torch.tensor(
            2,
            dtype=torch.long,
        ),
        key_ids=torch.tensor(
            [10, 11],
            dtype=torch.long,
        ),
        value_ids=torch.tensor(
            [20, 21],
            dtype=torch.long,
        ),
    )


def build_event() -> TensorizedEvent:
    return TensorizedEvent(
        event_token_id=torch.tensor(
            3,
            dtype=torch.long,
        ),
        key_ids=torch.tensor(
            [12, 13],
            dtype=torch.long,
        ),
        value_ids=torch.tensor(
            [22, 23],
            dtype=torch.long,
        ),
        calendar_features=torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            dtype=torch.float32,
        ),
        elapsed_time_feature=torch.tensor(
            0.7,
            dtype=torch.float32,
        ),
    )


def test_move_profile_to_device():
    device = get_test_device()

    profile = build_profile()

    moved_profile = move_profile_to_device(
        profile=profile,
        device=device,
    )

    assert moved_profile.user_token_id.device == device
    assert moved_profile.key_ids.device == device
    assert moved_profile.value_ids.device == device


def test_move_event_to_device():
    device = get_test_device()

    event = build_event()

    moved_event = move_event_to_device(
        event=event,
        device=device,
    )

    assert moved_event.event_token_id.device == device
    assert moved_event.key_ids.device == device
    assert moved_event.value_ids.device == device
    assert moved_event.calendar_features.device == device
    assert moved_event.elapsed_time_feature.device == device


def test_move_user_to_device_moves_nested_tensors():
    device = get_test_device()

    user = TensorizedUser(
        user_id=42,
        profile=build_profile(),
        events=(
            build_event(),
            build_event(),
        ),
    )

    moved_user = move_user_to_device(
        user=user,
        device=device,
    )

    assert moved_user.user_id == 42

    assert moved_user.profile.user_token_id.device == device
    assert moved_user.profile.key_ids.device == device
    assert moved_user.profile.value_ids.device == device

    for event in moved_user.events:
        assert event.event_token_id.device == device
        assert event.key_ids.device == device
        assert event.value_ids.device == device
        assert event.calendar_features.device == device
        assert event.elapsed_time_feature.device == device
