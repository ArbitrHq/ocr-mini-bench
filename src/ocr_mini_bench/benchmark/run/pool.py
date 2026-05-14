"""Async worker pool. Mirrors `src/benchmark/run/pool.ts`.

The TS implementation spins up `workerCount` workers that share a moving
cursor; each worker pulls the next index off the list and awaits the
runner. We mirror that exact shape rather than using `asyncio.Semaphore`
+ `gather` because the worker-pool style preserves task ordering across
workers and produces deterministic scheduling under fast/slow mixes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def run_in_pool(
    items: list[T],
    max_workers: int,
    runner: Callable[[T], Awaitable[None]],
) -> None:
    if not items:
        return

    worker_count = max(1, min(max_workers, len(items)))
    cursor = 0

    async def worker() -> None:
        nonlocal cursor
        while True:
            index = cursor
            cursor += 1
            if index >= len(items):
                return
            await runner(items[index])

    await asyncio.gather(*(worker() for _ in range(worker_count)))
