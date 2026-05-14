"""Mistral OCR provider. Mirrors `src/ocr/providers/mistral.ts`.

Two code paths:
- `mistral-ocr-*` models go straight to `/v1/ocr` with a JSON-schema document
  annotation; the response's `document_annotation` becomes the run text.
- All other Mistral models do OCR-then-chat: first `/v1/ocr` for markdown, then
  `/v1/chat/completions` to extract key/value pairs from that markdown.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from ...lib.type_guards import (
    get_array_property,
    get_number_property,
    get_record_property,
    get_string_property,
    is_record,
)
from ..constants import (
    MISTRAL_OCR_ANNOTATED_COST_PER_PAGE_USD,
    MISTRAL_OCR_COST_PER_PAGE_USD,
)
from ..provider_utils import is_mistral_ocr_model
from ..text_readers import (
    read_mistral_ocr_annotation_as_text,
    read_mistral_ocr_markdown,
    read_text_from_mistral_chat_response,
)
from ..types import OCRModelRunRequest, OCRModelRunResult

_ANNOTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "found": {"type": "boolean"},
                },
                "required": ["key", "value", "found"],
                "additionalProperties": False,
            },
        },
        "missing_keys": {
            "type": "array",
            "items": {"type": "string"},
        },
        "notes": {"type": "string"},
    },
    "required": ["pairs", "missing_keys"],
    "additionalProperties": False,
}


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    body: dict[str, Any],
    err_label: str,
) -> dict[str, Any]:
    response = await client.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
    )
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{err_label} request returned invalid response.") from exc
    if not is_record(data):
        raise RuntimeError(f"{err_label} request returned invalid response.")
    if response.status_code >= 400:
        error_obj = get_record_property(data, "error")
        message = get_string_property(error_obj, "message") if error_obj else None
        raise RuntimeError(message or f"{err_label} request failed.")
    return data


async def run_mistral_ocr(
    *,
    request: OCRModelRunRequest,
    api_key: str,
    started_at: float,
    max_output_tokens: int,
    client: httpx.AsyncClient,
) -> OCRModelRunResult:
    if is_mistral_ocr_model(request.model_id):
        annotation_prompt = f"{request.system_prompt}\n\n{request.user_prompt}".strip()
        ocr_data = await _post_json(
            client,
            "https://api.mistral.ai/v1/ocr",
            api_key,
            {
                "model": request.model_id,
                "document": {
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{request.pdf_base64}",
                },
                "document_annotation_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ocr_key_extraction",
                        "schema": _ANNOTATION_SCHEMA,
                    },
                },
                "document_annotation_prompt": annotation_prompt,
            },
            "Mistral OCR",
        )

        annotation_text = read_mistral_ocr_annotation_as_text(ocr_data)
        markdown_text = read_mistral_ocr_markdown(ocr_data)
        if annotation_text:
            response_text = annotation_text
        else:
            fallback: dict[str, Any] = {
                "pairs": [],
                "missing_keys": [],
                "notes": (
                    "No document annotation returned; markdown fallback included."
                    if markdown_text
                    else "No OCR output."
                ),
                "markdown": markdown_text,
            }
            response_text = json.dumps(fallback, separators=(",", ":"), ensure_ascii=False)

        usage_info = get_record_property(ocr_data, "usage_info")
        pages_processed = (
            int(get_number_property(usage_info, "pages_processed") or 0) if usage_info else 0
        )
        has_annotation = bool(annotation_text)
        per_page_cost = (
            MISTRAL_OCR_ANNOTATED_COST_PER_PAGE_USD
            if has_annotation
            else MISTRAL_OCR_COST_PER_PAGE_USD
        )
        total_cost_usd = max(0, pages_processed) * per_page_cost

        return OCRModelRunResult(
            text=response_text,
            input_tokens=0,
            output_tokens=0,
            latency_ms=int((time.time() - started_at) * 1000),
            cached_input_tokens=0,
            cache_hit=False,
            cache_write_tokens=0,
            total_cost_usd=total_cost_usd,
            no_cache_cost_usd=total_cost_usd,
        )

    ocr_data = await _post_json(
        client,
        "https://api.mistral.ai/v1/ocr",
        api_key,
        {
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{request.pdf_base64}",
            },
        },
        "Mistral OCR",
    )

    pages = get_array_property(ocr_data, "pages") or []
    md_chunks: list[str] = []
    for page in pages:
        if not is_record(page):
            continue
        md = get_string_property(page, "markdown") or ""
        if md:
            md_chunks.append(md)
    ocr_markdown = "\n\n".join(md_chunks)

    chat_data = await _post_json(
        client,
        "https://api.mistral.ai/v1/chat/completions",
        api_key,
        {
            "model": request.model_id,
            "temperature": 0,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {
                    "role": "user",
                    "content": f"{request.user_prompt}\n\nDocument OCR markdown:\n{ocr_markdown}",
                },
            ],
        },
        "Mistral chat",
    )

    usage = get_record_property(chat_data, "usage")
    input_tokens = int(get_number_property(usage, "prompt_tokens") or 0) if usage else 0
    output_tokens = int(get_number_property(usage, "completion_tokens") or 0) if usage else 0

    return OCRModelRunResult(
        text=read_text_from_mistral_chat_response(chat_data),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=int((time.time() - started_at) * 1000),
        cached_input_tokens=0,
        cache_hit=False,
        cache_write_tokens=0,
    )
