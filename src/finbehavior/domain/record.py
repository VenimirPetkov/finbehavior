from dataclasses import dataclass
from datetime import datetime

from .event import Event
from .profile import ProfileState


@dataclass
class UserRecord:
    user_id: int
    evaluation_point: datetime
    profile: ProfileState
    events: list[Event]

    def __post_init__(self):
        for event in self.events:
            if event.created > self.evaluation_point:
                raise ValueError("Event cannot occur after the evaluation point")
