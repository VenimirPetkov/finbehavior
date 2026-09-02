from dataclasses import dataclass

import torch

from finbehavior.models.masked_value_prediction_head import (
    MaskedValuePredictionHead,
)
from finbehavior.models.model import (
    FinBehaviorModel,
)
from finbehavior.tokenization.vocabulary import (
    Vocabulary,
)
from finbehavior.training.masked_value_batch_prediction import (
    predict_masked_value_batch,
)
from finbehavior.training.masking import (
    MaskedValueExample,
)


@dataclass(frozen=True)
class MaskedValuePrediction:
    token_id: int
    token: str
    probability: float


def decode_top_k_predictions(
    logits: torch.Tensor,
    vocabulary: Vocabulary,
    k: int = 5,
) -> tuple[MaskedValuePrediction, ...]:
    if logits.ndim != 1:
        raise ValueError("Logits must have shape " "[vocabulary_size]")

    if logits.shape[0] != len(vocabulary):
        raise ValueError("Logits vocabulary size does not " "match vocabulary")

    if k <= 0 or k > len(vocabulary):
        raise ValueError("k must be between 1 and " "vocabulary size")

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    top_k = torch.topk(
        probabilities,
        k=k,
    )

    return tuple(
        MaskedValuePrediction(
            token_id=token_id.item(),
            token=vocabulary.get_token(token_id.item()),
            probability=probability.item(),
        )
        for probability, token_id in zip(
            top_k.values,
            top_k.indices,
            strict=True,
        )
    )


def predict_masked_value_top_k(
    model: FinBehaviorModel,
    prediction_head: MaskedValuePredictionHead,
    example: MaskedValueExample,
    vocabulary: Vocabulary,
    k: int = 5,
) -> tuple[MaskedValuePrediction, ...]:
    model_was_training = model.training
    prediction_head_was_training = prediction_head.training

    model.eval()
    prediction_head.eval()

    try:
        with torch.no_grad():
            batch_prediction = predict_masked_value_batch(
                model=model,
                prediction_head=(prediction_head),
                examples=(example,),
            )

            return decode_top_k_predictions(
                logits=(batch_prediction.logits[0]),
                vocabulary=vocabulary,
                k=k,
            )
    finally:
        model.train(model_was_training)

        prediction_head.train(prediction_head_was_training)
