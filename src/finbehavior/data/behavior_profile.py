from dataclasses import dataclass


@dataclass
class BehaviorProfile:
    income_level: float
    spending_tendency: float
    travel_tendency: float
    investing_tendency: float
    app_activity: float
    communication_engagement: float

    def __post_init__(self):
        values = (
            self.income_level,
            self.spending_tendency,
            self.travel_tendency,
            self.investing_tendency,
            self.app_activity,
            self.communication_engagement,
        )

        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Behavior profile values must be between 0.0 and 1.0")
