import math
import random
from datetime import datetime

import torch

from time import perf_counter

from finbehavior.data.generators.dataset import (
    generate_dataset,
)
from finbehavior.domain.record import (
    UserRecord,
)
from finbehavior.models.factory import (
    build_finbehavior_model,
)
from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
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
from finbehavior.tokenization.config.numerical import (
    DEFAULT_NUMERICAL_BUCKET_COUNT,
)
from finbehavior.tokenization.fit import (
    fit_numerical_tokenization,
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
    build_vocabulary,
)
from finbehavior.training.masked_value_evaluation import (
    evaluate_masked_values,
)
from finbehavior.training.masked_value_training_loop import (
    train_masked_values,
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

NUMBER_OF_BUCKETS = DEFAULT_NUMERICAL_BUCKET_COUNT
EXAMPLES_PER_USER = 8
BATCH_SIZE = 64
TRAIN_BATCH_SHUFFLE_SEED = 321

TRAIN_EPOCH_COUNT = 5
LEARNING_RATE = 0.003

DATASET_SEED = 42
TRAIN_EXAMPLE_SELECTION_SEED = 123
VALIDATION_EXAMPLE_SELECTION_SEED = 124
TORCH_SEED = 0


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(
    device: torch.device,
) -> None:
    print()
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")


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
                    numerical_bucketizer=bucketizer,
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
            example = mask_event_value(
                user=user,
                event_index=event_index,
                field_index=field_index,
                mask_token_id=mask_token_id,
            )

            examples.append(example)

    return tuple(examples)


def print_dataset_summary(
    train_user_count: int,
    validation_user_count: int,
    train_example_count: int,
    validation_example_count: int,
) -> None:
    print()
    print(f"Train users: " f"{train_user_count}")
    print(f"Validation users: " f"{validation_user_count}")
    print(f"Train examples: " f"{train_example_count}")
    print(f"Validation examples: " f"{validation_example_count}")


def print_loss_header() -> None:
    print()
    print("Epoch | Train Loss | Validation Loss")
    print("------------------------------------")


def print_epoch_loss(
    epoch: int,
    train_loss: float,
    validation_loss: float,
) -> None:
    print(f"{epoch:>5} | " f"{train_loss:>10.4f} | " f"{validation_loss:>15.4f}")


def run_experiment() -> None:
    torch.manual_seed(TORCH_SEED)

    device = get_device()

    print_device_info(device)

    print()
    print("Generating synthetic users...")

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

    print("Fitting tokenizer " "on training users...")

    vocabulary = build_vocabulary()

    bucketizer = QuantileBucketizer(
        number_of_buckets=NUMBER_OF_BUCKETS,
    )

    fit_numerical_tokenization(
        records=split.train_records,
        bucketizer=bucketizer,
        vocabulary=vocabulary,
    )

    print("Tensorizing training users...")

    train_users = tensorize_records(
        records=split.train_records,
        vocabulary=vocabulary,
        bucketizer=bucketizer,
        device=device,
    )

    print("Tensorizing validation users...")

    validation_users = tensorize_records(
        records=split.validation_records,
        vocabulary=vocabulary,
        bucketizer=bucketizer,
        device=device,
    )

    mask_token_id = vocabulary.get_id(MASK_TOKEN)

    print("Sampling training examples...")

    train_examples = build_sampled_examples(
        users=train_users,
        mask_token_id=mask_token_id,
        examples_per_user=EXAMPLES_PER_USER,
        seed=TRAIN_EXAMPLE_SELECTION_SEED,
    )

    print("Sampling validation examples...")

    validation_examples = build_sampled_examples(
        users=validation_users,
        mask_token_id=mask_token_id,
        examples_per_user=EXAMPLES_PER_USER,
        seed=VALIDATION_EXAMPLE_SELECTION_SEED,
    )

    train_user_ids = {example.user.user_id for example in train_examples}

    validation_user_ids = {example.user.user_id for example in validation_examples}

    assert train_user_ids.isdisjoint(validation_user_ids)

    model = build_finbehavior_model(
        vocabulary_size=len(vocabulary),
    ).to(device)

    prediction_head = MaskedValuePredictionHead(
        vocabulary_size=len(vocabulary),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(prediction_head.parameters()),
        lr=LEARNING_RATE,
    )

    print_dataset_summary(
        train_user_count=len(split.train_records),
        validation_user_count=len(split.validation_records),
        train_example_count=len(train_examples),
        validation_example_count=len(validation_examples),
    )

    print()
    print("Measuring initial losses...")

    initial_train_loss = evaluate_masked_values(
        model=model,
        prediction_head=prediction_head,
        examples=train_examples,
        batch_size=BATCH_SIZE,
    )

    initial_validation_loss = evaluate_masked_values(
        model=model,
        prediction_head=prediction_head,
        examples=validation_examples,
        batch_size=BATCH_SIZE,
    )

    train_losses = [initial_train_loss]

    validation_losses = [initial_validation_loss]

    print_loss_header()

    print_epoch_loss(
        epoch=0,
        train_loss=initial_train_loss,
        validation_loss=(initial_validation_loss),
    )

    for epoch in range(
        1,
        TRAIN_EPOCH_COUNT + 1,
    ):
        training_start = perf_counter()

        train_masked_values(
            model=model,
            prediction_head=prediction_head,
            optimizer=optimizer,
            examples=train_examples,
            epoch_count=1,
            batch_size=BATCH_SIZE,
            shuffle_seed=(TRAIN_BATCH_SHUFFLE_SEED + epoch),
        )

        training_seconds = perf_counter() - training_start

        evaluation_start = perf_counter()

        train_loss = evaluate_masked_values(
            model=model,
            prediction_head=prediction_head,
            examples=train_examples,
            batch_size=BATCH_SIZE,
        )

        validation_loss = evaluate_masked_values(
            model=model,
            prediction_head=prediction_head,
            examples=validation_examples,
            batch_size=BATCH_SIZE,
        )
        evaluation_seconds = perf_counter() - evaluation_start

        print(
            f"Timing: training="
            f"{training_seconds:.2f}s, "
            f"evaluation="
            f"{evaluation_seconds:.2f}s"
        )

        train_losses.append(train_loss)

        validation_losses.append(validation_loss)

        print_epoch_loss(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
        )

    best_validation_epoch = min(
        range(len(validation_losses)),
        key=lambda index: (validation_losses[index]),
    )

    print()
    print(f"Best validation epoch: " f"{best_validation_epoch}")
    print(f"Best validation loss: " f"{validation_losses[best_validation_epoch]:.4f}")

    assert len(train_losses) == (TRAIN_EPOCH_COUNT + 1)

    assert len(validation_losses) == (TRAIN_EPOCH_COUNT + 1)

    assert train_losses[-1] < (train_losses[0])

    assert all(math.isfinite(loss) for loss in validation_losses)


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
