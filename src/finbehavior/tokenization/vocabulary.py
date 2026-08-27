from finbehavior.tokenization.categorical import (
    get_categorical_tokens,
)
from finbehavior.tokenization.keys import (
    get_key_tokens,
)

from .special_tokens import SPECIAL_TOKENS, UNK_TOKEN


class Vocabulary:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []

        for token in SPECIAL_TOKENS:
            self.add(token)

    def add(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]

        token_id = len(self._id_to_token)

        self._token_to_id[token] = token_id
        self._id_to_token.append(token)

        return token_id

    def get_id(self, token: str) -> int:
        return self._token_to_id[token]

    def encode(self, token: str) -> int:
        return self._token_to_id.get(
            token,
            self._token_to_id[UNK_TOKEN],
        )

    def get_token(self, token_id: int) -> str:
        return self._id_to_token[token_id]

    def __len__(self) -> int:
        return len(self._id_to_token)


def build_vocabulary() -> Vocabulary:
    vocab = Vocabulary()

    for token in get_key_tokens():
        vocab.add(token)

    for token in get_categorical_tokens():
        vocab.add(token)

    return vocab
