from dataclasses import dataclass
from datetime import datetime

from finbehavior.domain.types import FieldValue
from .enums import EventSource
    
@dataclass
class Event:
    created: datetime
    source: EventSource
    fields: dict[str, FieldValue]