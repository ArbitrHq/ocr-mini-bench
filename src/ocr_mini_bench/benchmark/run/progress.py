"""Progress reporter for the running benchmark. Mirrors
`src/benchmark/run/progress.ts`.

Produces stdout lines compatible with the TS reporter — same prefix,
same field ordering — so log scrapers don't break across the port.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ProgressEvent:
    document_id: str
    model_label: str
    run_number: int
    ok: bool
    error: str | None = None


def create_progress_reporter(
    *,
    total_tasks: int,
    expected_runs_by_document: dict[str, int],
    started_at_ms: int,
) -> Callable[[ProgressEvent], None]:
    completed_by_document: dict[str, int] = {}
    counters = {"completed": 0, "ok": 0, "fail": 0}
    log_every = max(1, total_tasks // 20)

    def report(event: ProgressEvent) -> None:
        counters["completed"] += 1
        if event.ok:
            counters["ok"] += 1
        else:
            counters["fail"] += 1

        doc_completed = completed_by_document.get(event.document_id, 0) + 1
        completed_by_document[event.document_id] = doc_completed

        completed = counters["completed"]
        percent = f"{(completed / total_tasks) * 100:.1f}" if total_tasks > 0 else "100.0"
        elapsed_sec = f"{(time.time() * 1000 - started_at_ms) / 1000:.1f}"
        expected_for_doc = expected_runs_by_document.get(event.document_id, 0)
        document_done = expected_for_doc > 0 and doc_completed == expected_for_doc
        periodic_tick = (
            completed == 1 or completed == total_tasks or completed % log_every == 0
        )

        if not (periodic_tick or document_done or not event.ok):
            return

        doc_suffix = (
            f" | document complete: {event.document_id} ({doc_completed}/{expected_for_doc})"
            if document_done
            else ""
        )
        error_suffix = (
            f" | last error: {event.error or 'unknown error'}" if not event.ok else ""
        )
        print(
            f"[ocr-benchmark] {completed}/{total_tasks} ({percent}%) | "
            f"ok:{counters['ok']} fail:{counters['fail']} | elapsed:{elapsed_sec}s | "
            f"model:{event.model_label} run:{event.run_number}{doc_suffix}{error_suffix}"
        )

    return report
