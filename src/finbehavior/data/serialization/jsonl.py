import json
from collections.abc import Iterable
from pathlib import Path

from finbehavior.data.synthetic_user import SyntheticUser

from .user import synthetic_user_from_dict, synthetic_user_to_dict


def write_dataset_jsonl(
    users: Iterable[SyntheticUser],
    path: str | Path,
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for user in users:
            json.dump(
                synthetic_user_to_dict(user),
                file,
                ensure_ascii=False,
            )

            file.write("\n")


def read_dataset_jsonl(path: str | Path) -> list[SyntheticUser]:
    input_path = Path(path)
    users: list[SyntheticUser] = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error

            if not isinstance(data, dict):
                raise TypeError(f"JSONL line {line_number} must contain an object")

            try:
                users.append(synthetic_user_from_dict(data))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid synthetic user on line {line_number}: {error}"
                ) from error

    return users
