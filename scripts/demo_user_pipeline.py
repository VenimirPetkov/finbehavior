from collections import Counter
from datetime import datetime

from finbehavior.data.generators.dataset import generate_dataset
from finbehavior.data.reference.field_keys import AMOUNT_FIELD
from finbehavior.data.synthetic_user import SyntheticUser
from finbehavior.domain.enums import EventSource
from finbehavior.tokenization.fit import fit_numerical_tokenization
from finbehavior.tokenization.keys import get_event_key_token
from finbehavior.tokenization.numerical import QuantileBucketizer
from finbehavior.tokenization.types import TokenizedUser
from finbehavior.tokenization.user import tokenize_user_record
from finbehavior.tokenization.vocabulary import (
    Vocabulary,
    build_vocabulary,
)

DEMO_NUMBER_OF_USERS = 20
DEMO_TRAINING_USER_COUNT = 15
DEMO_SEED = 42
DEMO_EVENT_DISPLAY_LIMIT = 5

DEMO_START = datetime(
    2026,
    1,
    1,
)

DEMO_EVALUATION_POINT = datetime(
    2026,
    2,
    1,
)


def main() -> None:
    users = generate_dataset(
        number_of_users=DEMO_NUMBER_OF_USERS,
        start=DEMO_START,
        evaluation_point=DEMO_EVALUATION_POINT,
        seed=DEMO_SEED,
    )

    training_users = users[:DEMO_TRAINING_USER_COUNT]

    held_out_users = users[DEMO_TRAINING_USER_COUNT:]

    training_records = [user.record for user in training_users]

    vocabulary = build_vocabulary()
    bucketizer = QuantileBucketizer()

    fit_numerical_tokenization(
        records=training_records,
        bucketizer=bucketizer,
        vocabulary=vocabulary,
    )

    demo_user = held_out_users[0]

    tokenized_user = tokenize_user_record(
        record=demo_user.record,
        vocabulary=vocabulary,
        numerical_bucketizer=bucketizer,
    )

    print_dataset_summary(
        users=users,
        training_users=training_users,
        held_out_users=held_out_users,
    )

    print_learned_boundaries(bucketizer)

    print_raw_user(demo_user)

    print_tokenized_user(
        raw_user=demo_user,
        tokenized_user=tokenized_user,
        vocabulary=vocabulary,
    )


def print_dataset_summary(
    users: list[SyntheticUser],
    training_users: list[SyntheticUser],
    held_out_users: list[SyntheticUser],
) -> None:
    source_counts = Counter(
        event.source for user in users for event in user.record.events
    )

    total_events = sum(source_counts.values())

    print()
    print("DATASET")
    print("=" * 60)

    print(f"users:              {len(users)}")

    print(f"training users:     {len(training_users)}")

    print(f"held-out users:     {len(held_out_users)}")

    print(f"total events:       {total_events}")

    print()
    print("events by source:")

    for source in EventSource:
        print(f"  {source.value:<15} " f"{source_counts[source]}")


def print_learned_boundaries(
    bucketizer: QuantileBucketizer,
) -> None:
    print()
    print("LEARNED NUMERICAL BOUNDARIES")
    print("=" * 60)

    for source in (
        EventSource.TRANSACTION,
        EventSource.TRADING,
    ):
        key_token = get_event_key_token(
            source,
            AMOUNT_FIELD,
        )

        boundaries = bucketizer.get_boundaries(key_token)

        print()
        print(key_token)

        for index, boundary in enumerate(boundaries):
            print(f"  boundary_{index}: " f"{boundary:.2f}")


def print_raw_user(
    user: SyntheticUser,
) -> None:
    print()
    print("HELD-OUT SYNTHETIC USER")
    print("=" * 60)

    print(f"user_id: " f"{user.record.user_id}")

    print(f"events:  " f"{len(user.record.events)}")

    print()
    print("HIDDEN GENERATOR BEHAVIOR " "(NOT MODEL INPUT)")

    for name, value in vars(user.behavior).items():
        print(f"  {name:<28} " f"{value:.3f}")

    print()
    print("RAW PROFILE")

    for key, value in user.record.profile.fields.items():
        print(f"  {key:<28} " f"{value}")


def print_tokenized_user(
    raw_user: SyntheticUser,
    tokenized_user: TokenizedUser,
    vocabulary: Vocabulary,
) -> None:
    print()
    print("TOKENIZED PROFILE")
    print("=" * 60)

    user_token = vocabulary.get_token(tokenized_user.profile.user_token_id)

    print(f"{user_token} -> " f"{tokenized_user.profile.user_token_id}")

    for field in tokenized_user.profile.fields:
        key_token = vocabulary.get_token(field.key_id)

        value_token = vocabulary.get_token(field.value_id)

        print()
        print(f"  {key_token:<35} " f"-> {field.key_id}")

        print(f"  {value_token:<35} " f"-> {field.value_id}")

    print()
    print("TOKENIZED HISTORY")
    print("=" * 60)

    for index, (
        raw_event,
        tokenized_event,
    ) in enumerate(
        zip(
            raw_user.record.events,
            tokenized_user.events,
        ),
        start=1,
    ):
        if index > DEMO_EVENT_DISPLAY_LIMIT:
            break

        print()
        print(f"EVENT {index}")

        print(f"  time:   " f"{raw_event.created}")

        print(f"  source: " f"{raw_event.source.value}")

        event_token = vocabulary.get_token(tokenized_event.event_token_id)

        print(f"  {event_token} -> " f"{tokenized_event.event_token_id}")

        print()
        print("  raw:")

        for key, value in raw_event.fields.items():
            print(f"    {key:<30} " f"{value}")

        print()
        print("  tokenized:")

        for field in tokenized_event.fields:
            key_token = vocabulary.get_token(field.key_id)

            value_token = vocabulary.get_token(field.value_id)

            print(f"    {key_token:<30} " f"-> {field.key_id}")

            print(f"    {value_token:<30} " f"-> {field.value_id}")

        print()
        print("  elapsed feature: " f"{tokenized_event.elapsed_time_feature:.6f}")

    hidden_events = len(tokenized_user.events) - DEMO_EVENT_DISPLAY_LIMIT

    if hidden_events > 0:
        print()
        print(f"... {hidden_events} more events " "tokenized successfully")

    print()
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print(f"user_id:           " f"{tokenized_user.user_id}")

    print(f"profile fields:    " f"{len(tokenized_user.profile.fields)}")

    print(f"tokenized events:  " f"{len(tokenized_user.events)}")

    print(f"vocabulary size:   " f"{len(vocabulary)}")

    print()
    print("Synthetic data -> fitted tokenizer " "-> model-ready user")


if __name__ == "__main__":
    main()
