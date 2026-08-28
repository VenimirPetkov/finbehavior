import torch

from finbehavior.tokenization.types import (
    TokenizedEvent,
)

from .types import TensorizedEvent


def tensorize_event(
    event: TokenizedEvent,
) -> TensorizedEvent:
    key_ids = torch.tensor(
        [field.key_id for field in event.fields],
        dtype=torch.long,
    )

    value_ids = torch.tensor(
        [field.value_id for field in event.fields],
        dtype=torch.long,
    )

    event_token_id = torch.tensor(
        event.event_token_id,
        dtype=torch.long,
    )

    calendar_features = torch.tensor(
        event.calendar_features,
        dtype=torch.float32,
    )

    elapsed_time_feature = torch.tensor(
        event.elapsed_time_feature,
        dtype=torch.float32,
    )

    return TensorizedEvent(
        event_token_id=event_token_id,
        key_ids=key_ids,
        value_ids=value_ids,
        calendar_features=calendar_features,
        elapsed_time_feature=elapsed_time_feature,
    )
