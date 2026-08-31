from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class SequenceBatch:
    sequences: torch.Tensor
    attention_mask: torch.Tensor


def build_sequence_batch(
    sequences: tuple[torch.Tensor, ...],
) -> SequenceBatch:
    if not sequences:
        raise ValueError("Sequence batch must contain at least one sequence")

    sequence_lengths = torch.tensor(
        [sequence.shape[0] for sequence in sequences],
        device=sequences[0].device,
    )

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
    )

    max_sequence_length = padded_sequences.shape[1]

    positions = torch.arange(
        max_sequence_length,
        device=padded_sequences.device,
    ).unsqueeze(0)

    attention_mask = positions < sequence_lengths.unsqueeze(1)

    return SequenceBatch(
        sequences=padded_sequences,
        attention_mask=attention_mask,
    )
