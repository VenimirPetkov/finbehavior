from dataclasses import dataclass

from finbehavior.data.behavior_profile import BehaviorProfile
from finbehavior.domain.record import UserRecord


@dataclass
class SyntheticUser:
    behavior: BehaviorProfile
    record: UserRecord
