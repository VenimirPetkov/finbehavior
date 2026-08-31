from dataclasses import replace

from finbehavior.training.example_batching import (
    build_length_aware_batches,
)


def test_length_aware_batches_group_similar_lengths(
    masked_value_example_factory,
):
    base_example = masked_value_example_factory(10)

    event = base_example.user.events[0]

    def with_event_count(
        event_count: int,
    ):
        user = replace(
            base_example.user,
            events=tuple(event for _ in range(event_count)),
        )

        return replace(
            base_example,
            user=user,
        )

    examples = (
        with_event_count(10),
        with_event_count(2),
        with_event_count(9),
        with_event_count(1),
    )

    batches = build_length_aware_batches(
        examples=examples,
        batch_size=2,
    )

    event_counts = tuple(
        tuple(len(example.user.events) for example in batch) for batch in batches
    )

    assert event_counts == (
        (1, 2),
        (9, 10),
    )


def test_length_aware_batches_shuffle_deterministically(
    masked_value_example_factory,
):
    examples = tuple(
        masked_value_example_factory(key_id)
        for key_id in range(
            10,
            30,
        )
    )

    first_batches = build_length_aware_batches(
        examples=examples,
        batch_size=4,
        shuffle=True,
        seed=123,
    )

    second_batches = build_length_aware_batches(
        examples=examples,
        batch_size=4,
        shuffle=True,
        seed=123,
    )

    different_seed_batches = build_length_aware_batches(
        examples=examples,
        batch_size=4,
        shuffle=True,
        seed=124,
    )

    def get_key_order(
        batches,
    ):
        return tuple(
            example.user.events[0].key_ids.item()
            for batch in batches
            for example in batch
        )

    first_order = get_key_order(first_batches)

    second_order = get_key_order(second_batches)

    different_order = get_key_order(different_seed_batches)

    assert first_order == second_order

    assert first_order != different_order
