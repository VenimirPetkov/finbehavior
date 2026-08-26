
from enum import Enum


class EventSource(str, Enum):
    TRANSACTION = "transaction"
    APP = "app"
    TRADING = "trading"
    COMMUNICATION = "communication"
