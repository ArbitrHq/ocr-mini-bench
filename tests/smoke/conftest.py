"""Load `py/.env` (and the repo-root `.env` as a fallback) so smoke tests can
pick up provider keys without an explicit `set -a; . .env` step.

Scoped to the `smoke/` directory only — unit and replay tests never touch
network or keys, and we don't want their behavior to depend on local env files.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_PY_ENV = Path(__file__).resolve().parents[2] / ".env"
_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"

if _PY_ENV.exists():
    load_dotenv(_PY_ENV, override=False)
if _ROOT_ENV.exists():
    load_dotenv(_ROOT_ENV, override=False)
