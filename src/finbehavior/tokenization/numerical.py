from bisect import bisect_right
from collections.abc import Sequence
from math import isfinite
from statistics import quantiles

from .config.numerical import (
    DEFAULT_NUMERICAL_BUCKET_COUNT,
    MIN_NUMERICAL_BUCKET_COUNT,
    MIN_VALUES_FOR_QUANTILES,
)

ZERO_BUCKET_NAME = "zero"


class QuantileBucketizer:
    def __init__(
        self,
        number_of_buckets: int = DEFAULT_NUMERICAL_BUCKET_COUNT,
    ) -> None:
        if number_of_buckets < MIN_NUMERICAL_BUCKET_COUNT:
            raise ValueError("Number of buckets must be at least 2")

        self.number_of_buckets = number_of_buckets
        self._boundaries: dict[str, tuple[float, ...]] = {}

    def fit(
        self,
        key_token: str,
        values: Sequence[int | float],
    ) -> None:
        numeric_values = tuple(float(value) for value in values)

        if not all(isfinite(value) for value in numeric_values):
            raise ValueError("Numerical values must be finite")

        non_zero_values = tuple(value for value in numeric_values if value != 0.0)

        if len(non_zero_values) < MIN_VALUES_FOR_QUANTILES:
            raise ValueError(
                "At least two non-zero values are required "
                "to fit quantile boundaries"
            )

        boundaries = quantiles(
            non_zero_values,
            n=self.number_of_buckets,
            method="inclusive",
        )

        self._boundaries[key_token] = tuple(boundaries)

    def transform(
        self,
        key_token: str,
        value: int | float,
    ) -> str:
        if key_token not in self._boundaries:
            raise ValueError(f"No quantile boundaries fitted for key: {key_token}")

        numeric_value = float(value)

        if not isfinite(numeric_value):
            raise ValueError("Numerical value must be finite")

        if numeric_value == 0.0:
            return self._build_zero_bucket_token(key_token)

        boundaries = self._boundaries[key_token]

        bucket_index = bisect_right(
            boundaries,
            numeric_value,
        )

        return self._build_bucket_token(
            key_token=key_token,
            bucket_index=bucket_index,
        )

    def get_boundaries(
        self,
        key_token: str,
    ) -> tuple[float, ...]:
        if key_token not in self._boundaries:
            raise ValueError(f"No quantile boundaries fitted for key: {key_token}")

        return self._boundaries[key_token]

    def get_bucket_tokens(
        self,
        key_token: str,
    ) -> tuple[str, ...]:
        if key_token not in self._boundaries:
            raise ValueError(f"No quantile boundaries fitted for key: {key_token}")

        quantile_tokens = tuple(
            self._build_bucket_token(
                key_token=key_token,
                bucket_index=bucket_index,
            )
            for bucket_index in range(self.number_of_buckets)
        )

        return (
            self._build_zero_bucket_token(key_token),
            *quantile_tokens,
        )

    @staticmethod
    def _build_bucket_token(
        key_token: str,
        bucket_index: int,
    ) -> str:
        return f"{key_token}.bucket_{bucket_index}"

    @staticmethod
    def _build_zero_bucket_token(
        key_token: str,
    ) -> str:
        return f"{key_token}.{ZERO_BUCKET_NAME}"
