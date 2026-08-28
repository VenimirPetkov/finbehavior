import json
from pathlib import Path

from .numerical import QuantileBucketizer
from .vocabulary import Vocabulary

TOKENIZER_STATE_VERSION = 1


def save_tokenizer(
    path: Path,
    vocabulary: Vocabulary,
    bucketizer: QuantileBucketizer,
) -> None:
    state = {
        "version": TOKENIZER_STATE_VERSION,
        "vocabulary": list(vocabulary.get_tokens()),
        "numerical": {
            "number_of_buckets": (bucketizer.number_of_buckets),
            "boundaries": {
                key: list(boundaries)
                for key, boundaries in bucketizer.get_all_boundaries().items()
            },
        },
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            state,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_tokenizer(
    path: Path,
) -> tuple[Vocabulary, QuantileBucketizer]:
    state = json.loads(path.read_text(encoding="utf-8"))

    version = state["version"]

    if version != TOKENIZER_STATE_VERSION:
        raise ValueError(f"Unsupported tokenizer state version: " f"{version}")

    vocabulary = Vocabulary.from_tokens(state["vocabulary"])

    numerical_state = state["numerical"]

    bucketizer = QuantileBucketizer.from_boundaries(
        number_of_buckets=numerical_state["number_of_buckets"],
        boundaries=numerical_state["boundaries"],
    )

    return vocabulary, bucketizer
