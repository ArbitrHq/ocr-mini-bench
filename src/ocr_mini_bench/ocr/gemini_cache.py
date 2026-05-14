"""Gemini explicit-cache helpers. Mirrors `src/ocr/gemini-cache.ts`.

Builds Gemini `generateContent` payloads, optionally referencing a server-side
`cachedContents` entry for models in `GEMINI_EXPLICIT_CACHE_MODELS`. The cache
entries are keyed by `(modelId, sha256(system_prompt + pdf_base64))` and reused
within a single process while their TTL is alive.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime
from typing import Any

import httpx

from .constants import (
    GEMINI_CACHE_TTL_SECONDS,
    GEMINI_EXPLICIT_CACHE_MODELS,
    GeminiCacheEntry,
)

_cache_by_key: dict[str, GeminiCacheEntry] = {}
_in_flight: dict[str, asyncio.Task[str | None]] = {}


def should_use_gemini_explicit_cache(model_id: str) -> bool:
    return model_id in GEMINI_EXPLICIT_CACHE_MODELS


def build_gemini_thinking_config(model_id: str) -> dict[str, int] | None:
    if "flash" in model_id.lower():
        return {"thinkingBudget": 0}
    return None


def _build_gemini_cache_key(model_id: str, system_prompt: str, pdf_base64: str) -> str:
    h = hashlib.sha256()
    h.update(system_prompt.encode("utf-8"))
    h.update(b"\n")
    h.update(pdf_base64.encode("utf-8"))
    return f"{model_id}:{h.hexdigest()}"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _parse_expire_time(value: str) -> int:
    # Gemini returns RFC3339 like "2026-05-14T12:00:00.123Z".
    try:
        normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
        return int(datetime.fromisoformat(normalized).timestamp() * 1000)
    except (ValueError, TypeError):
        return _now_ms() + GEMINI_CACHE_TTL_SECONDS * 1000


async def _get_or_create_gemini_cache_name(
    *,
    client: httpx.AsyncClient,
    api_key: str,
    model_id: str,
    system_prompt: str,
    pdf_base64: str,
) -> str | None:
    cache_key = _build_gemini_cache_key(model_id, system_prompt, pdf_base64)
    now = _now_ms()
    existing = _cache_by_key.get(cache_key)
    if existing and existing.expire_at_ms > now + 5000:
        return existing.name

    in_flight = _in_flight.get(cache_key)
    if in_flight is not None:
        return await in_flight

    async def _create() -> str | None:
        response = await client.post(
            "https://generativelanguage.googleapis.com/v1beta/cachedContents",
            params={"key": api_key},
            json={
                "model": f"models/{model_id}",
                "displayName": f"ocr-{cache_key[:16]}",
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "application/pdf",
                                    "data": pdf_base64,
                                }
                            }
                        ],
                    }
                ],
                "ttl": f"{GEMINI_CACHE_TTL_SECONDS}s",
            },
        )
        try:
            data = response.json()
        except ValueError:
            return None
        if response.status_code >= 400 or not isinstance(data, dict):
            return None
        name = data.get("name")
        if not isinstance(name, str):
            return None
        expire_time = data.get("expireTime")
        expire_at_ms = (
            _parse_expire_time(expire_time)
            if isinstance(expire_time, str)
            else _now_ms() + GEMINI_CACHE_TTL_SECONDS * 1000
        )
        _cache_by_key[cache_key] = GeminiCacheEntry(name=name, expire_at_ms=expire_at_ms)
        return name

    task: asyncio.Task[str | None] = asyncio.create_task(_create())
    _in_flight[cache_key] = task
    try:
        return await task
    finally:
        _in_flight.pop(cache_key, None)


async def build_gemini_generate_payload(
    *,
    client: httpx.AsyncClient,
    api_key: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    pdf_base64: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    thinking_config = build_gemini_thinking_config(model_id)

    if should_use_gemini_explicit_cache(model_id):
        cache_name = await _get_or_create_gemini_cache_name(
            client=client,
            api_key=api_key,
            model_id=model_id,
            system_prompt=system_prompt,
            pdf_base64=pdf_base64,
        )
        if cache_name:
            generation_config: dict[str, Any] = {
                "maxOutputTokens": max_output_tokens,
                "temperature": 0,
            }
            if thinking_config is not None:
                generation_config["thinkingConfig"] = thinking_config
            return {
                "cachedContent": cache_name,
                "contents": [
                    {"role": "user", "parts": [{"text": user_prompt}]},
                ],
                "generationConfig": generation_config,
            }

    generation_config = {
        "maxOutputTokens": max_output_tokens,
        "temperature": 0,
    }
    if thinking_config is not None:
        generation_config["thinkingConfig"] = thinking_config
    return {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "application/pdf",
                            "data": pdf_base64,
                        }
                    },
                    {"text": user_prompt},
                ],
            }
        ],
        "generationConfig": generation_config,
    }


def _reset_caches_for_tests() -> None:
    _cache_by_key.clear()
    _in_flight.clear()
