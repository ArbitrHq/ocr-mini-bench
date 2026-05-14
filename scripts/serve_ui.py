"""Tiny HTTP server for the OCR mini-bench UI. Python port of
`scripts/serve_ui.mjs`, using stdlib `http.server` to avoid a FastAPI dep
for a static-file + JSON-endpoint app.

Routes:
    GET /api/meta         - paths the server resolved
    GET /api/leaderboard  - latest leaderboard.frontend.json (or fallbacks)
    GET /api/debug        - latest latest.debug.json (or fallback)
    GET /api/document?source_pdf=PATH - serve a PDF from bench_documents
    GET /                 - ui/index.html
    GET /debug            - ui/debug.html
    GET /<file>           - static files under ui/

Usage:
    uv run python scripts/serve_ui.py [--port=4173]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

from ocr_mini_bench.config.paths import PATHS

REPO_ROOT = PATHS.dataset.manifest.parent.parent
UI_DIR = REPO_ROOT / "ui"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def _walk_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def _latest_by_name(filename: str) -> Path | None:
    matches = [p for p in _walk_files(ARTIFACTS_DIR) if p.name == filename]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _discover_artifact_paths() -> dict[str, Path | None]:
    leaderboard = (
        _latest_by_name("leaderboard.frontend.json")
        or _latest_by_name("latest.frontend.json")
        or _latest_by_name("latest.json")
    )
    debug = _latest_by_name("latest.debug.json") or _latest_by_name("snapshot.debug.json")
    return {"leaderboardPath": leaderboard, "debugPath": debug}


_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".pdf": "application/pdf",
}


def _content_type_for(file_path: Path) -> str:
    return _CONTENT_TYPES.get(file_path.suffix.lower(), "application/octet-stream")


def _safe_resolve_repo_path(input_path: str) -> Path | None:
    """Resolve `input_path` against REPO_ROOT, refusing any result that
    escapes the repo. Path traversal guard."""
    if not input_path:
        return None
    abs_path = (REPO_ROOT / input_path).resolve()
    try:
        abs_path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return None
    return abs_path


class _Handler(BaseHTTPRequestHandler):
    artifacts: ClassVar[dict[str, Path | None]] = {}

    def log_message(self, format: str, *args: object) -> None:
        # Quieter default than the stdlib's stderr-per-request spam.
        return

    def _send_json(self, status: int, payload: object) -> None:
        body = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, body: str, content_type: str = "text/plain; charset=utf-8") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, file_path: Path) -> None:
        try:
            data = file_path.read_bytes()
        except OSError:
            self._send_text(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", _content_type_for(file_path))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_raw_json(self, file_path: Path) -> None:
        try:
            data = file_path.read_bytes()
        except OSError:
            self._send_json(404, {"error": "Read failed."})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path or "/")
            pathname = parsed.path

            if pathname == "/api/meta":
                self._send_json(
                    200,
                    {
                        "repo_root": str(REPO_ROOT),
                        "artifacts_dir": str(ARTIFACTS_DIR),
                        "leaderboard_path": str(self.artifacts.get("leaderboardPath"))
                        if self.artifacts.get("leaderboardPath")
                        else None,
                        "debug_path": str(self.artifacts.get("debugPath"))
                        if self.artifacts.get("debugPath")
                        else None,
                    },
                )
                return

            if pathname == "/api/leaderboard":
                p = self.artifacts.get("leaderboardPath")
                if not p:
                    self._send_json(404, {"error": "No leaderboard artifact found."})
                    return
                self._send_raw_json(p)
                return

            if pathname == "/api/debug":
                p = self.artifacts.get("debugPath")
                if not p:
                    self._send_json(404, {"error": "No debug artifact found."})
                    return
                self._send_raw_json(p)
                return

            if pathname == "/api/document":
                params = parse_qs(parsed.query)
                source_pdf = params.get("source_pdf", [""])[0]
                target = _safe_resolve_repo_path(source_pdf)
                if not target:
                    self._send_json(400, {"error": "Invalid source_pdf path."})
                    return
                if not target.exists():
                    self._send_json(404, {"error": f"Document not found: {source_pdf}"})
                    return
                self._send_file(target)
                return

            if pathname in ("/", "/index.html"):
                self._send_file(UI_DIR / "index.html")
                return
            if pathname in ("/debug", "/debug.html"):
                self._send_file(UI_DIR / "debug.html")
                return

            ui_candidate = (UI_DIR / pathname.lstrip("/")).resolve()
            try:
                ui_candidate.relative_to(UI_DIR.resolve())
            except ValueError:
                self._send_text(404, "Not found")
                return
            if ui_candidate.exists() and ui_candidate.is_file():
                self._send_file(ui_candidate)
                return

            self._send_text(404, "Not found")
        except Exception as e:
            self._send_json(500, {"error": str(e) or "Internal server error."})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--port", type=int, default=4173)
    args = parser.parse_args(argv)

    artifacts = _discover_artifact_paths()
    _Handler.artifacts = artifacts

    server = HTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"OCR mini-bench UI available at http://127.0.0.1:{args.port}")
    print(f"Leaderboard source: {artifacts['leaderboardPath'] or 'not found'}")
    print(f"Debug source: {artifacts['debugPath'] or 'not found'}")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
