# Run from repo root — uv handles the virtualenv automatically.

default: test

sync:
    uv sync --all-extras

test:
    uv run pytest

test-unit:
    uv run pytest -m unit

test-replay:
    uv run pytest -m replay

lint:
    uv run ruff check src tests scripts

format:
    uv run ruff format src tests scripts

typecheck:
    uv run mypy src

check: lint typecheck test
