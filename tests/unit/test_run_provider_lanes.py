"""Tests for run.provider_lanes — per-provider serial execution under a
shared concurrency cap on distinct providers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from ocr_mini_bench.benchmark.run.provider_lanes import run_by_provider_lanes


@dataclass
class _Task:
    provider: str
    label: str


@pytest.mark.unit
async def test_tasks_within_same_provider_run_serially() -> None:
    in_flight_by_provider: dict[str, int] = {}
    high_water_by_provider: dict[str, int] = {}
    lock = asyncio.Lock()

    async def runner(task: _Task) -> None:
        async with lock:
            in_flight_by_provider[task.provider] = (
                in_flight_by_provider.get(task.provider, 0) + 1
            )
            high_water_by_provider[task.provider] = max(
                high_water_by_provider.get(task.provider, 0),
                in_flight_by_provider[task.provider],
            )
        await asyncio.sleep(0.01)
        async with lock:
            in_flight_by_provider[task.provider] -= 1

    tasks = [
        _Task("google", "a"),
        _Task("google", "b"),
        _Task("openai", "c"),
        _Task("openai", "d"),
    ]
    await run_by_provider_lanes(tasks, 4, runner)

    for provider, hw in high_water_by_provider.items():
        assert hw == 1, f"{provider} ran concurrently: high water {hw}"


@pytest.mark.unit
async def test_distinct_providers_run_in_parallel_under_cap() -> None:
    """Two distinct providers should run concurrently when the lane cap allows."""
    cross_provider_high_water = 0
    in_flight = 0
    lock = asyncio.Lock()

    async def runner(task: _Task) -> None:
        nonlocal in_flight, cross_provider_high_water
        async with lock:
            in_flight += 1
            cross_provider_high_water = max(cross_provider_high_water, in_flight)
        await asyncio.sleep(0.02)
        async with lock:
            in_flight -= 1

    tasks = [_Task("a", "1"), _Task("b", "2")]
    await run_by_provider_lanes(tasks, 2, runner)
    assert cross_provider_high_water == 2


@pytest.mark.unit
async def test_lane_cap_constrains_provider_count() -> None:
    cross_provider_high_water = 0
    in_flight = 0
    lock = asyncio.Lock()

    async def runner(task: _Task) -> None:
        nonlocal in_flight, cross_provider_high_water
        async with lock:
            in_flight += 1
            cross_provider_high_water = max(cross_provider_high_water, in_flight)
        await asyncio.sleep(0.02)
        async with lock:
            in_flight -= 1

    tasks = [_Task("a", "1"), _Task("b", "2"), _Task("c", "3")]
    await run_by_provider_lanes(tasks, 2, runner)
    # Three providers, lane cap 2 → at most 2 concurrent.
    assert cross_provider_high_water == 2


@pytest.mark.unit
async def test_within_provider_order_preserved() -> None:
    order: list[tuple[str, str]] = []

    async def runner(task: _Task) -> None:
        order.append((task.provider, task.label))

    tasks = [
        _Task("openai", "1"),
        _Task("openai", "2"),
        _Task("openai", "3"),
        _Task("google", "1"),
        _Task("google", "2"),
    ]
    await run_by_provider_lanes(tasks, 4, runner)
    openai_seq = [label for provider, label in order if provider == "openai"]
    google_seq = [label for provider, label in order if provider == "google"]
    assert openai_seq == ["1", "2", "3"]
    assert google_seq == ["1", "2"]
