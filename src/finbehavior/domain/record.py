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