import random
from datetime import datetime
from pathlib import Path

import torch

from finbehavior.data.generators.dataset import (
    generate_dataset,
)
from finbehavior.domain.record import (
    UserRecord,
)
from finbehavior.inference.masked_value import (
    predict_masked_value_top_k,
)
from finbehavior.persistence.checkpoint import (
    checkpoint_exists,
    load_checkpoint,
)
from finbehavior.tensorization.device import (
    move_user_to_device,
)
from finbehavior.tensorization.types import (
    TensorizedUser,
)
from finbehavior.tensorization.user import (
    tensorize_user,
)
from finbehavior.tokenization.numerical import (
    QuantileBucketizer,
)
from finbehavior.tokenization.special_tokens import (
    MASK_TOKEN,
)
from finbehavior.tokenization.user import (
    tokenize_user_record,
)
from finbehavior.tokenization.vocabulary import (
    Vocabulary,
)
from finbehavior.training.masking import (
    MaskedValueExample,
    mask_event_value,
)
from finbehavior.training.user_split import (
    split_user_records,
)

DATASET_USER_COUNT = 1024
TRAIN_FRACTION = 0.8

DATASET_SEED = 42
VALIDATION_EXAMPLE_SELECTION_SEED = 124

EXAMPLES_PER_USER = 8
DEMO_EXAMPLE_INDEX = 0
TOP_K = 5

BEST_CHECKPOINT_DIRECTORY = Path("checkpoints/generalization_best")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensorize_records(
    records: tuple[UserRecord, ...],
    vocabulary: Vocabulary,
    bucketizer: QuantileBucketizer,
    device: torch.device,
) -> tuple[TensorizedUser, ...]:
    return tuple(
        move_user_to_device(
            user=tensorize_user(
                tokenize_user_record(
                    record=record,
                    vocabulary=vocabulary,
                    numerical_bucketizer=(bucketizer),
                )
            ),
            device=device,
        )
        for record in records
    )


def get_maskable_positions(
    user: TensorizedUser,
) -> list[tuple[int, int]]:
    return [
        (
            event_index,
            field_index,
        )
        for event_index, event in enumerate(user.events)
        for field_index in range(event.value_ids.numel())
    ]


def build_sampled_examples(
    users: tuple[TensorizedUser, ...],
    mask_token_id: int,
    examples_per_user: int,
    seed: int,
) -> tuple[MaskedValueExample, ...]:
    rng = random.Random(seed)

    examples = []

    for user in users:
        maskable_positions = get_maskable_positions(user)

        if not maskable_positions:
            continue

        selection_count = min(
            examples_per_user,
            len(maskable_positions),
        )

        selected_positions = rng.sample(
            maskable_positions,
            selection_count,
        )

        for (
            event_index,
            field_index,
        ) in selected_positions:
            examples.append(
                mask_event_value(
                    user=user,
                    event_index=event_index,
                    field_index=field_index,
                    mask_token_id=(mask_token_id),
                )
            )

    return tuple(examples)


def main() -> None:
    device = get_device()

    print()
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: " f"{torch.cuda.get_device_name(0)}")

    if not checkpoint_exists(BEST_CHECKPOINT_DIRECTORY):
        raise FileNotFoundError(
            "Best checkpoint not found at " f"{BEST_CHECKPOINT_DIRECTORY}"
        )

    print()
    print("Loading best checkpoint...")

    checkpoint = load_checkpoint(
        directory=(BEST_CHECKPOINT_DIRECTORY),
        device=device,
    )

    print(f"Checkpoint epoch: " f"{checkpoint.epoch}")
    print(f"Validation loss: " f"{checkpoint.validation_loss:.4f}")
    print(f"Top-1 accuracy: " f"{checkpoint.top_1_accuracy * 100:.1f}%")
    print(f"Top-5 accuracy: " f"{checkpoint.top_5_accuracy * 100:.1f}%")

    print()
    print("Rebuilding deterministic " "validation dataset...")

    synthetic_users = generate_dataset(
        number_of_users=DATASET_USER_COUNT,
        start=datetime(
            2026,
            1,
            1,
        ),
        evaluation_point=datetime(
            2026,
            1,
            8,
        ),
        seed=DATASET_SEED,
    )

    records = tuple(synthetic_user.record for synthetic_user in synthetic_users)

    split = split_user_records(
        records=records,
        train_fraction=TRAIN_FRACTION,
        seed=DATASET_SEED,
    )

    validation_users = tensorize_records(
        records=split.validation_records,
        vocabulary=checkpoint.vocabulary,
        bucketizer=checkpoint.bucketizer,
        device=device,
    )

    mask_token_id = checkpoint.vocabulary.get_id(MASK_TOKEN)

    validation_examples = build_sampled_examples(
        users=validation_users,
        mask_token_id=mask_token_id,
        examples_per_user=(EXAMPLES_PER_USER),
        seed=(VALIDATION_EXAMPLE_SELECTION_SEED),
    )

    if not validation_examples:
        raise RuntimeError("No validation examples found")

    if DEMO_EXAMPLE_INDEX >= len(validation_examples):
        raise IndexError("Demo example index is outside " "the validation examples")

    example = validation_examples[DEMO_EXAMPLE_INDEX]

    vocabulary = checkpoint.vocabulary

    masked_event = example.user.events[example.event_index]

    event_token = vocabulary.get_token(masked_event.event_token_id.item())

    field_token = vocabulary.get_token(masked_event.key_ids[example.field_index].item())

    true_token = vocabulary.get_token(example.target_token_id.item())

    predictions = predict_masked_value_top_k(
        model=checkpoint.model,
        prediction_head=(checkpoint.prediction_head),
        example=example,
        vocabulary=vocabulary,
        k=TOP_K,
    )

    print()
    print("==============================")
    print("FinBehavior Inference Demo")
    print("==============================")

    print(f"Validation user: " f"{example.user.user_id}")
    print(f"Event index: " f"{example.event_index}")
    print(f"Event type: " f"{event_token}")
    print(f"Masked field: " f"{field_token}")

    print()
    print(f"True value: " f"{true_token}")

    print()
    print(f"Top-{TOP_K} predictions:")

    predicted_tokens = []

    for rank, prediction in enumerate(
        predictions,
        start=1,
    ):
        predicted_tokens.append(prediction.token)

        true_marker = "  <-- TRUE" if prediction.token == true_token else ""

        print(
            f"{rank}. "
            f"{prediction.token} "
            f"({prediction.probability * 100:.2f}%)"
            f"{true_marker}"
        )

    print()

    if true_token == predictions[0].token:
        print("Result: TOP-1 HIT")
    elif true_token in predicted_tokens:
        print("Result: TOP-5 HIT")
    else:
        print("Result: MISS")


if __name__ == "__main__":
    main()
