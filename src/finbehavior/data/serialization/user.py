from dataclasses import asdict

from finbehavior.data.synthetic_user import SyntheticUser
from finbehavior.domain.event import Event


def event_to_dict(event: Event) -> dict[str, object]:
    return {
        "created": event.created.isoformat(),
        "source": event.source.value,
        "fields": event.fields,
    }


def synthetic_user_to_dict(
    user: SyntheticUser,
) -> dict[str, object]:
    return {
        "behavior": asdict(user.behavior),
        "record": {
            "user_id": user.record.user_id,
            "evaluation_point": (
                user.record.evaluation_point.isoformat()
            ),
            "profile": user.record.profile.fields,
            "events": [
                event_to_dict(event)
                for event in user.record.events
            ],
        },
    }