from finbehavior.data.reference.profile import (
    BALANCE_QUANTILE_VALUES,
)


def get_balance_quantile_token(
    value: int,
) -> str:
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError("Balance quantile must be an integer")

    if value not in BALANCE_QUANTILE_VALUES:
        raise ValueError(f"Invalid balance quantile: {value}")

    return f"balance_quantile_{value}"


def get_balance_quantile_tokens() -> tuple[str, ...]:
    return tuple(get_balance_quantile_token(value) for value in BALANCE_QUANTILE_VALUES)
