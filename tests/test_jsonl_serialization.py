import json
from datetime import datetime

from finbehavior.data.generators.dataset import generate_dataset
from finbehavior.data.serialization.jsonl import (
    write_dataset_jsonl,
)


def test_write_dataset_jsonl(tmp_path):
    users = generate_dataset(
        number_of_users=2,
        start=datetime(2026, 1, 1),
        evaluation_point=datetime(2026, 2, 1),
        seed=42,
    )

    output_path = tmp_path / "dataset.jsonl"

    write_dataset_jsonl(
        users=users,
        path=output_path,
    )

    assert output_path.exists()

    lines = output_path.read_text(
        encoding="utf-8",
    ).splitlines()

    assert len(lines) == 2

    first_user = json.loads(lines[0])
    second_user = json.loads(lines[1])

    assert first_user["record"]["user_id"] == 0
    assert second_user["record"]["user_id"] == 1
