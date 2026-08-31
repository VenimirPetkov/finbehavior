import random

from .masking import MaskedValueExample

PROFILE_SEQUENCE_LENGTH = 1


def get_example_sequence_length(
    example: MaskedValueExample,
) -> int:
    return len(example.user.events) + PROFILE_SEQUENCE_LENGTH


def build_length_aware_batches(
    examples: tuple[MaskedValueExample, ...],
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
) -> tuple[
    tuple[MaskedValueExample, ...],
    ...,
]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    if not examples:
        return ()

    ordered_examples = list(examples)

    if shuffle:
        rng = random.Random(seed)

        rng.shuffle(ordered_examples)
    else:
        rng = None

    ordered_examples.sort(key=get_example_sequence_length)

    batches = [
        tuple(ordered_examples[start_index : start_index + batch_size])
        for start_index in range(
            0,
            len(ordered_examples),
            batch_size,
        )
    ]

    if rng is not None:
        rng.shuffle(batches)

    return tuple(batches)
