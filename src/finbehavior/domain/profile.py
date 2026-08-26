from dataclasses import dataclass

ProfileValue = str | int | float | bool

@dataclass
class ProfileState:
    fields: dict[str, ProfileValue]