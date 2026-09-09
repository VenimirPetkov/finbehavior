from dataclasses import asdict
from datetime import datetime
from collections.abc import Mapping

from finbehavior.data.behavior_profile import BehaviorProfile
from finbehavior.data.synthetic_user import SyntheticUser
from finbehavior.domain.enums import EventSource
from finbehavior.domain.event import Event
from finbehavior.domain.profile import ProfileState
from finbehavior.domain.record import UserRecord


def event_to_dict(event: Event) -> dict[str, object]:
    return {
        "created": event.created.isoformat(),
        "source": event.source.value,
        "fields": event.fields,
    }


def event_from_dict(data: Mapping[str, object]) -> Event:
    created = data.get("created")
    source = data.get("source")
    fields = data.get("fields")

    if not isinstance(created, str):
        raise TypeError("Event 'created' must be an ISO datetime string")

    if not isinstance(source, str):
        raise TypeError("Event 'source' must be a string")

    if not isinstance(fields, Mapping):
        raise TypeError("Event 'fields' must be an object")

    parsed_fields: dict[str, str | int | float | bool] = {}

    for key, value in fields.items():
        if not isinstance(key, str):
            raise TypeError("Event field names must be strings")

        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"Event field '{key}' has an unsupported value")

        parsed_fields[key] = value

    return Event(
        created=datetime.fromisoformat(created),
        source=EventSource(source),
        fields=parsed_fields,
    )


def user_record_to_dict(record: UserRecord) -> dict[str, object]:
    return {
        "user_id": record.user_id,
        "evaluation_point": record.evaluation_point.isoformat(),
        "profile": dict(record.profile.fields),
        "events": [event_to_dict(event) for event in record.events],
    }


def user_record_from_dict(data: Mapping[str, object]) -> UserRecord:
    user_id = data.get("user_id")
    evaluation_point = data.get("evaluation_point")
    profile = data.get("profile")
    events = data.get("events")

    if isinstance(user_id, bool) or not isinstance(user_id, int):
        raise TypeError("Record 'user_id' must be an integer")

    if not isinstance(evaluation_point, str):
        raise TypeError("Record 'evaluation_point' must be an ISO datetime string")

    if not isinstance(profile, Mapping):
        raise TypeError("Record 'profile' must be an object")

    if not isinstance(events, list):
        raise TypeError("Record 'events' must be a list")

    parsed_profile: dict[str, str | int | float | bool] = {}

    for key, value in profile.items():
        if not isinstance(key, str):
            raise TypeError("Profile field names must be strings")

        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"Profile field '{key}' has an unsupported value")

        parsed_profile[key] = value

    parsed_events = []

    for event in events:
        if not isinstance(event, Mapping):
            raise TypeError("Every record event must be an object")

        parsed_events.append(event_from_dict(event))

    return UserRecord(
        user_id=user_id,
        evaluation_point=datetime.fromisoformat(evaluation_point),
        profile=ProfileState(fields=parsed_profile),
        events=parsed_events,
    )


def synthetic_user_to_dict(
    user: SyntheticUser,
) -> dict[str, object]:
    return {
        "behavior": asdict(user.behavior),
        "record": user_record_to_dict(user.record),
    }


def synthetic_user_from_dict(data: Mapping[str, object]) -> SyntheticUser:
    behavior = data.get("behavior")
    record = data.get("record")

    if not isinstance(behavior, Mapping):
        raise TypeError("Synthetic user 'behavior' must be an object")

    if not isinstance(record, Mapping):
        raise TypeError("Synthetic user 'record' must be an object")

    behavior_fields = (
        "income_level",
        "spending_tendency",
        "travel_tendency",
        "investing_tendency",
        "app_activity",
        "communication_engagement",
    )
    behavior_values: dict[str, float] = {}

    for field_name in behavior_fields:
        value = behavior.get(field_name)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Behavior field '{field_name}' must be a number")

        behavior_values[field_name] = float(value)

    return SyntheticUser(
        behavior=BehaviorProfile(**behavior_values),
        record=user_record_from_dict(record),
    )
