from .special_tokens import SPECIAL_TOKENS


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

    def get_token(self, token_id: int) -> str:
        return self._id_to_token[token_id]

    def __len__(self) -> int:
        return len(self._id_to_token)