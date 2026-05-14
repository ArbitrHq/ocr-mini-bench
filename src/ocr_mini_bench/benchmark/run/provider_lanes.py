"""Provider-lane scheduling. Mirrors `src/benchmark/run/provider-lanes.ts`.

Groups tasks by `provider`, sorts queues by provider name (locale-compare
in TS == lexicographic on ASCII), then dispatches one queue per worker so
each provider's tasks execute serially while distinct providers can run
in parallel up to `max_provider_lanes`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeVar

from .pool import run_in_pool


class _HasProvider(Protocol):
    provider: str


T = TypeVar("T", bound=_HasProvider)


async def run_by_provider_lanes(
    tasks: Sequence[T],
    max_provider_lanes: int,
    runner: Callable[[T], Awaitable[None]],
) -> None:
    by_provider: dict[str, list[T]] = {}
    for task in tasks:
        by_provider.setdefault(task.provider, []).append(task)

    provider_queues = [
        (provider, by_provider[provider]) for provider in sorted(by_provider.keys())
    ]

    async def queue_runner(entry: tuple[str, list[T]]) -> None:
        _, queue_tasks = entry
        for task in queue_tasks:
            await runner(task)

    await run_in_pool(provider_queues, max_provider_lanes, queue_runner)
