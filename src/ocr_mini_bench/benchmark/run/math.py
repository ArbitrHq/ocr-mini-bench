"""Stats helpers used by leaderboard aggregation. Mirrors
`src/benchmark/run/math.ts`.

`percentile` matches the TS "nearest-rank" implementation and `round` uses
JS-style round-half-away-from-zero, because both rules show up *in the
artifacts themselves* (rank tables, formatted percentages). Plain `sum` is
used elsewhere — Python's pairwise summation is more accurate than JS's
left-fold and the difference only shows up at the float-precision boundary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def std_dev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = max(0.0, sum((value - avg) ** 2 for value in values) / len(values))
    return math.sqrt(variance)


def pct(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return (part / total) * 100


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile, matching the TS reference.

    `rank = clamp(ceil(p/100 * n) - 1, 0, n - 1)`.
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = min(len(sorted_values) - 1, max(0, math.ceil((p / 100) * len(sorted_values)) - 1))
    return sorted_values[rank]


def round_half_away_from_zero(value: float, decimals: int = 4) -> float:
    """JS `Math.round(value * 10**decimals) / 10**decimals` — rounds halves
    toward +infinity for positive numbers and toward -infinity for negatives.
    Python's `round` uses banker's rounding, which produces different values
    at half-points; this helper preserves the TS contract."""
    precision: int = 10**decimals
    scaled = value * precision
    rounded = int(math.floor(scaled + 0.5) if scaled >= 0 else -math.floor(-scaled + 0.5))
    return rounded / precision
