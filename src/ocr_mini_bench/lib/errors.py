"""Error message helpers."""

from __future__ import annotations


def to_error_message(error: object) -> str:
    """Extract a human-readable error message from an arbitrary error value.

    Mirrors TS `toErrorMessage`: returns `.message` for Error-like objects,
    `str(error)` for already-stringified exceptions, and a fixed fallback
    otherwise. The fallback string is part of the contract — surfaced in
    `raw.jsonl` rows for failed runs.
    """
    if isinstance(error, BaseException):
        message = str(error)
        return message if message else "Model run failed."
    return "Model run failed."
