import torch

from finbehavior.tensorization.types import (
    TensorizedEvent,
    TensorizedProfile,
    TensorizedUser,
)
from finbehavior.tokenization.config.temporal import (
    CALENDAR_FEATURE_DIMENSION,
)
from finbehavior.training.masked_value_examples import (
    build_masked_value_examples,
)


def build_event(
    value_ids: list[int],
) -> TensorizedEvent:
    return TensorizedEvent(
        event_token_id=torch.tensor(
            3,
            dtype=torch.long,
        ),
        key_ids=torch.tensor(
            [10] * len(value_ids),
            dtype=torch.long,
        ),
        value_ids=torch.tensor(
            value_ids,
            dtype=torch.long,
        ),
        calendar_features=torch.zeros(
            CALENDAR_FEATURE_DIMENSION,
            dtype=torch.float32,
        ),
        elapsed_time_feature=torch.tensor(
            0.5,
            dtype=torch.float32,
        ),
    )


def build_user(
    user_id: int,
    events: tuple[TensorizedEvent, ...],
) -> TensorizedUser:
    profile = TensorizedProfile(
        user_token_id=torch.tensor(
            2,
            dtype=torch.long,
        ),
        key_ids=torch.empty(
            0,
            dtype=torch.long,
        ),
        value_ids=torch.empty(
            0,
            dtype=torch.long,
        ),
    )

    return TensorizedUser(
        user_id=user_id,
        profile=profile,
        events=events,
    )


def test_build_masked_value_examples_masks_every_event_value():
    users = (
        build_user(
            user_id=1,
            events=(
                build_event([20, 21]),
                build_event([22]),
            ),
        ),
        build_user(
            user_id=2,
            events=(build_event([23, 24]),),
        ),
    )

    examples = build_masked_value_examples(
        users=users,
        mask_token_id=0,
    )

    assert len(examples) == 5

    targets = tuple(example.target_token_id.item() for example in examples)

    assert targets == (
        20,
        21,
        22,
        23,
        24,
    )

    positions = tuple(
        (
            example.user.user_id,
            example.event_index,
            example.field_index,
        )
        for example in examples
    )

    assert positions == (
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (2, 0, 0),
        (2, 0, 1),
    )


def test_build_masked_value_examples_keeps_original_users_unchanged():
    user = build_user(
        user_id=1,
        events=(build_event([20, 21]),),
    )

    original_values = user.events[0].value_ids.clone()

    build_masked_value_examples(
        users=(user,),
        mask_token_id=0,
    )

    assert torch.equal(
        user.events[0].value_ids,
        original_values,
    )
