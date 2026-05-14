"""Tests for run.math — guards the rounding/percentile contract that feeds
the leaderboard artifacts."""

from __future__ import annotations

import pytest

from ocr_mini_bench.benchmark.run.math import (
    mean,
    pct,
    percentile,
    round_half_away_from_zero,
    std_dev,
)


@pytest.mark.unit
class TestMean:
    def test_basic(self) -> None:
        assert mean([1, 2, 3, 4, 5]) == 3.0

    def test_empty(self) -> None:
        assert mean([]) == 0.0


@pytest.mark.unit
class TestStdDev:
    def test_zero_for_single_value(self) -> None:
        assert std_dev([7]) == 0.0

    def test_population_stddev(self) -> None:
        # TS uses population variance (divides by n, not n-1)
        # [2,4,4,4,5,5,7,9] → mean=5, var=4, stddev=2
        assert std_dev([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.0)


@pytest.mark.unit
class TestPct:
    def test_basic(self) -> None:
        assert pct(3, 4) == 75.0

    def test_zero_total(self) -> None:
        assert pct(1, 0) == 0.0

    def test_negative_total(self) -> None:
        assert pct(1, -5) == 0.0


@pytest.mark.unit
class TestPercentile:
    def test_empty(self) -> None:
        assert percentile([], 50) == 0.0

    def test_nearest_rank(self) -> None:
        # values = [10,20,30,40,50,60,70,80,90,100], n=10
        # p=95 → ceil(9.5)-1 = 9 → 100
        assert percentile([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 95) == 100
        # p=50 → ceil(5.0)-1 = 4 → 50
        assert percentile([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 50) == 50

    def test_p0_clamps_to_zero_index(self) -> None:
        # p=0 → ceil(0)-1 = -1 → clamped to 0
        assert percentile([10, 20, 30], 0) == 10


@pytest.mark.unit
class TestRoundHalfAwayFromZero:
    def test_basic(self) -> None:
        assert round_half_away_from_zero(1.23456, 2) == 1.23

    def test_half_rounds_away_from_zero(self) -> None:
        # Python's built-in round(0.5) == 0 (banker's); JS Math.round(0.5) == 1
        assert round_half_away_from_zero(0.5, 0) == 1.0
        assert round_half_away_from_zero(2.5, 0) == 3.0
        assert round_half_away_from_zero(-0.5, 0) == -1.0

    def test_default_decimals_is_four(self) -> None:
        assert round_half_away_from_zero(1.23456789) == 1.2346
