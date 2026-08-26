from dataclasses import dataclass

from .types import ScalarValue


@dataclass
class ProfileState:
    fields: dict[str, ScalarValue]
