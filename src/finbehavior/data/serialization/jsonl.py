import json
from collections.abc import Iterable
from pathlib import Path

from finbehavior.data.synthetic_user import SyntheticUser

from .user import synthetic_user_to_dict


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