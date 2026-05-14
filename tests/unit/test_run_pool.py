"""Tests for run.pool — async worker-pool concurrency primitive."""

from __future__ import annotations

import asyncio

import pytest

from ocr_mini_bench.benchmark.run.pool import run_in_pool


@pytest.mark.unit
async def test_empty_list_is_noop() -> None:
    calls: list[int] = []

    async def runner(item: int) -> None:
        calls.append(item)

    await run_in_pool([], 4, runner)
    assert calls == []


@pytest.mark.unit
async def test_processes_all_items_with_more_workers_than_items() -> None:
    items = [1, 2, 3]
    results: list[int] = []

    async def runner(item: int) -> None:
        results.append(item)

    await run_in_pool(items, 10, runner)
    assert sorted(results) == sorted(items)


@pytest.mark.unit
async def test_respects_max_workers() -> None:
    """Concurrency must be capped at max_workers — track the high-water mark."""
    items = list(range(10))
    in_flight = 0
    high_water = 0
    lock = asyncio.Lock()

    async def runner(item: int) -> None:
        nonlocal in_flight, high_water
        async with lock:
            in_flight += 1
            high_water = max(high_water, in_flight)
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1

    await run_in_pool(items, 3, runner)
    assert high_water <= 3


@pytest.mark.unit
async def test_workers_drain_shared_cursor() -> None:
    """A slow worker shouldn't block fast workers from picking up tasks."""
    items = list(range(20))
    handled: list[int] = []
    handled_lock = asyncio.Lock()

    async def runner(item: int) -> None:
        # Items 0..4 are slow; the remaining items should be picked up by
        # other workers off the shared cursor.
        await asyncio.sleep(0.05 if item < 5 else 0.001)
        async with handled_lock:
            handled.append(item)

    await run_in_pool(items, 4, runner)
    assert sorted(handled) == items


@pytest.mark.unit
async def test_max_workers_floored_at_one() -> None:
    items = [1, 2, 3]
    results: list[int] = []

    async def runner(item: int) -> None:
        results.append(item)

    await run_in_pool(items, 0, runner)
    assert results == items
