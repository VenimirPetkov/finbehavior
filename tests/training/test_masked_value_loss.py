import torch

from finbehavior.training.masked_value_loss import (
    masked_value_loss,
)


def test_masked_value_loss_returns_scalar():
    logits = torch.tensor(
        [0.2, 0.5, 3.0, -0.4],
    )

    target_token_id = torch.tensor(
        2,
        dtype=torch.long,
    )

    loss = masked_value_loss(
        logits,
        target_token_id,
    )

    assert loss.ndim == 0


def test_masked_value_loss_is_lower_for_correct_prediction():
    target_token_id = torch.tensor(
        2,
        dtype=torch.long,
    )

    good_logits = torch.tensor(
        [0.1, 0.2, 5.0, 0.3],
    )

    bad_logits = torch.tensor(
        [5.0, 0.2, 0.1, 0.3],
    )

    good_loss = masked_value_loss(
        good_logits,
        target_token_id,
    )

    bad_loss = masked_value_loss(
        bad_logits,
        target_token_id,
    )

    assert good_loss < bad_loss
