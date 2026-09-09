import torch

from finbehavior.models.positional_encoding import (
    SinusoidalPositionalEncoding,
)


def test_sinusoidal_positional_encoding_is_deterministic_and_position_specific():
    positional_encoding = SinusoidalPositionalEncoding(
        embedding_dimension=5,
    )

    sequence = torch.zeros(
        3,
        5,
    )

    first_result = positional_encoding(sequence)
    second_result = positional_encoding(sequence)

    assert torch.equal(
        first_result,
        second_result,
    )

    assert torch.equal(
        first_result[0],
        torch.tensor(
            [
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
            ]
        ),
    )

    assert not torch.equal(
        first_result[1],
        first_result[2],
    )


def test_sinusoidal_positional_encoding_has_no_checkpoint_state():
    positional_encoding = SinusoidalPositionalEncoding()

    assert positional_encoding.state_dict() == {}
