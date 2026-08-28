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
) -> TokenizedUser:
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

    latest_event_time = max(event.created for event in record.events)

    tokenized_events = tuple(
        tokenize_event(
            event=event,
            latest_event_time=latest_event_time,
            vocabulary=vocabulary,
            numerical_bucketizer=numerical_bucketizer,
        )
        for event in record.events
    )

    return TokenizedUser(
        user_id=record.user_id,
        profile=tokenized_profile,
        events=tokenized_events,
    )
