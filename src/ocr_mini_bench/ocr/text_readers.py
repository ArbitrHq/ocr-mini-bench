"""Provider-response text extraction. Mirrors `src/ocr/text-readers.ts`.

These helpers consume raw decoded JSON from each provider and produce the
canonical `text` field stored on `OCRModelRunResult`. They are deliberately
permissive: schema drift in any provider response must degrade to "" rather
than raise.
"""

from __future__ import annotations

import json
from typing import Any

from ..lib.type_guards import (
    get_array_property,
    get_record_property,
    get_string_property,
    is_array,
    is_record,
    is_string,
)
from .parsing import parse_first_json_object


def _compact_json_dumps(value: Any) -> str:
    # Match `JSON.stringify(value)` — no whitespace, no trailing newline.
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def read_mistral_ocr_annotation_as_text(data: dict[str, Any]) -> str:
    raw = data.get("document_annotation")
    if is_string(raw):
        parsed = parse_first_json_object(raw)
        return _compact_json_dumps(parsed) if parsed is not None else raw
    if is_record(raw):
        return _compact_json_dumps(raw)
    return ""


def read_mistral_ocr_markdown(data: dict[str, Any]) -> str:
    pages = get_array_property(data, "pages") or []
    chunks: list[str] = []
    for page in pages:
        if not is_record(page):
            continue
        md = get_string_property(page, "markdown") or ""
        if md:
            chunks.append(md)
    return "\n\n".join(chunks)


def read_text_from_anthropic_content(content: list[Any]) -> str:
    """Concatenate `text` blocks from an Anthropic Messages.Message.content list.

    Accepts both SDK objects (with `.type`/`.text` attributes) and plain dicts
    (as appear in recorded fixtures or raw HTTP).
    """
    chunks: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type != "text":
            continue
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "\n".join(chunks).strip()


def read_text_from_openai_response(data: dict[str, Any]) -> str:
    output_text = get_string_property(data, "output_text")
    if output_text and output_text.strip():
        return output_text.strip()

    output = get_array_property(data, "output") or []
    chunks: list[str] = []

    for item in output:
        if not is_record(item):
            continue
        content = get_array_property(item, "content") or []
        for part in content:
            if not is_record(part):
                continue
            if get_string_property(part, "type") == "output_text":
                text = get_string_property(part, "text")
                if text:
                    chunks.append(text)

    return "\n".join(chunks).strip()


def read_text_from_gemini_response(data: dict[str, Any]) -> str:
    candidates = get_array_property(data, "candidates") or []
    first_candidate = candidates[0] if candidates and is_record(candidates[0]) else None
    content = get_record_property(first_candidate, "content") if first_candidate else None
    parts = get_array_property(content, "parts") if content else None
    parts = parts or []
    chunks: list[str] = []
    for part in parts:
        if not is_record(part):
            continue
        text = get_string_property(part, "text") or ""
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def read_text_from_mistral_chat_response(data: dict[str, Any]) -> str:
    choices = get_array_property(data, "choices") or []
    first_choice = choices[0] if choices and is_record(choices[0]) else None
    message = get_record_property(first_choice, "message") if first_choice else None

    if not message:
        return ""

    content = message.get("content")

    if is_string(content):
        return content.strip()

    if is_array(content):
        chunks: list[str] = []
        for part in content:
            if not is_record(part):
                continue
            text = get_string_property(part, "text") or ""
            if text:
                chunks.append(text)
        return "\n".join(chunks).strip()

    return ""
