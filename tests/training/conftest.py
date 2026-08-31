from collections.abc import Callable

import pytest
import torch
from torch import nn

from finbehavior.models.config.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
)
from finbehavior.tensorization.types import (
    TensorizedEvent,
    TensorizedProfile,
    TensorizedUser,
)
from finbehavior.tokenization.config.temporal import (
    CALENDAR_FEATURE_DIMENSION,
)
from finbehavior.training.masking import (
    MaskedValueExample,
    mask_event_value,
)

MaskedValueExampleFactory = Callable[
    [int],
    MaskedValueExample,
]


class TrainableStubFinBehaviorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        initial_representation = torch.zeros(
            DEFAULT_EMBEDDING_DIMENSION,
        )

        initial_representation[0] = 1.0

        self.event_representation = nn.Parameter(initial_representation)

    def encode_sequence(
        self,
        user: TensorizedUser,
    ) -> torch.Tensor:
        profile_representation = torch.zeros_like(self.event_representation)

        return torch.stack(
            (
                profile_representation,
                self.event_representation,
            ),
            dim=0,
        )

    def encode_users(
        self,
        users: tuple[TensorizedUser, ...],
    ) -> torch.Tensor:
        return torch.stack(
            tuple(self.encode_sequence(user) for user in users),
            dim=0,
        )


@pytest.fixture
def masked_value_example_factory() -> MaskedValueExampleFactory:
    def build_masked_value_example(
        key_id: int,
    ) -> MaskedValueExample:
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

        event = TensorizedEvent(
            event_token_id=torch.tensor(
                3,
                dtype=torch.long,
            ),
            key_ids=torch.tensor(
                [key_id],
                dtype=torch.long,
            ),
            value_ids=torch.tensor(
                [2],
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

        user = TensorizedUser(
            user_id=1,
            profile=profile,
            events=(event,),
        )

        return mask_event_value(
            user=user,
            event_index=0,
            field_index=0,
            mask_token_id=0,
        )

    return build_masked_value_example


@pytest.fixture
def trainable_stub_model() -> TrainableStubFinBehaviorModel:
    return TrainableStubFinBehaviorModel()
