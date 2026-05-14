"""File-IO helpers for postprocess artifacts. Mirrors
`src/postprocess/io.ts`.

Critical for parity: TS uses 2-space indent + trailing newline on JSON
files and per-line + trailing newline on JSONL files. Match exactly.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def file_exists(target: str | Path) -> bool:
    return Path(target).exists()


def ensure_parent_dir(target: str | Path) -> None:
    Path(target).parent.mkdir(parents=True, exist_ok=True)


def read_json_file(target: str | Path) -> Any:
    return json.loads(Path(target).read_text(encoding="utf-8"))


def write_json_file(target: str | Path, value: Any) -> None:
    """Write JSON with 2-space indent + trailing newline (TS-compatible)."""
    path = Path(target)
    ensure_parent_dir(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False)
    path.write_text(payload + "\n", encoding="utf-8")


def write_json_lines_file(target: str | Path, rows: Iterable[Any]) -> None:
    """Write one JSON object per line. Trailing newline only if non-empty,
    matching the TS `payload.length > 0` guard."""
    path = Path(target)
    ensure_parent_dir(path)
    encoded = [json.dumps(row, ensure_ascii=False) for row in rows]
    payload = "\n".join(encoded)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def read_json_lines_file(target: str | Path) -> list[Any]:
    text = Path(target).read_text(encoding="utf-8")
    out: list[Any] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            out.append(json.loads(stripped))
        except json.JSONDecodeError:
            # Skip malformed lines, matching TS behavior.
            continue
    return out


def timestamp_for_filename(moment: datetime | None = None) -> str:
    when = moment if moment is not None else datetime.now(UTC)
    utc = when.astimezone(UTC)
    iso = utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"
    return iso.replace(":", "-").replace(".", "-")
