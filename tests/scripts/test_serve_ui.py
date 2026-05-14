"""Tests for `scripts/serve_ui.py`. Boots the server on an ephemeral port
and exercises each route via urllib."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "serve_ui.py"


def _wait_for_server(url: str, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.5).read()
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.1)
    raise RuntimeError(f"server did not come up at {url}")


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), f"--port={port}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_for_server(f"{base}/api/meta")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def _get(url: str) -> tuple[int, bytes, str]:
    try:
        resp = urllib.request.urlopen(url, timeout=2)
        return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


@pytest.mark.unit
def test_meta_endpoint(server: str) -> None:
    status, body, ctype = _get(f"{server}/api/meta")
    assert status == 200
    assert "application/json" in ctype
    payload = json.loads(body)
    assert "repo_root" in payload
    assert "artifacts_dir" in payload
    assert "leaderboard_path" in payload
    assert "debug_path" in payload


@pytest.mark.unit
def test_path_traversal_blocked(server: str) -> None:
    status, body, _ = _get(f"{server}/api/document?source_pdf=../../etc/passwd")
    assert status == 400
    assert json.loads(body) == {"error": "Invalid source_pdf path."}


@pytest.mark.unit
def test_missing_document_404(server: str) -> None:
    status, body, _ = _get(f"{server}/api/document?source_pdf=does-not-exist.pdf")
    assert status == 404
    payload = json.loads(body)
    assert "Document not found" in payload["error"]


@pytest.mark.unit
def test_index_html_served(server: str) -> None:
    status, body, ctype = _get(f"{server}/")
    if status == 404:
        pytest.skip("ui/index.html not present in this checkout")
    assert status == 200
    assert "text/html" in ctype
    assert b"<html" in body.lower() or b"<!doctype" in body.lower()


@pytest.mark.unit
def test_unknown_route_404(server: str) -> None:
    status, body, _ = _get(f"{server}/does-not-exist")
    assert status == 404
    assert body == b"Not found"
