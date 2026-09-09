from finbehavior.domain.record import UserRecord

from .event import tokenize_event
from .numerical import QuantileBucketizer
from .profile import tokenize_profile
from .types import TokenizedUser
from .vocabulary import Vocabulary


def tokenize_user_record(
    record: UserRecord,
    vocabulary: Vocabulary,
    numerical_bucketizer: QuantileBucketizer,
    max_events: int | None = None,
) -> TokenizedUser:
    if max_events is not None and max_events < 1:
        raise ValueError("max_events must be at least 1")

    tokenized_profile = tokenize_profile(
        profile=record.profile,
        vocabulary=vocabulary,
    )

    if not record.events:
        return TokenizedUser(
            user_id=record.user_id,
            profile=tokenized_profile,
            events=(),
        )

    chronological_events = sorted(record.events, key=lambda event: event.created)

    if max_events is not None:
        chronological_events = chronological_events[-max_events:]

    latest_event_time = chronological_events[-1].created

    tokenized_events = tuple(
        tokenize_event(
            event=event,
            latest_event_time=latest_event_time,
            vocabulary=vocabulary,
            numerical_bucketizer=numerical_bucketizer,
        )
        for event in chronological_events
    )

    return TokenizedUser(
        user_id=record.user_id,
        profile=tokenized_profile,
        events=tokenized_events,
    )
