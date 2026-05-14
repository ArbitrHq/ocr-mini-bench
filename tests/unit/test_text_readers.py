"""Unit tests for `ocr.text_readers`. Mirrors implicit TS behavior in
`src/ocr/text-readers.ts`.
"""

from __future__ import annotations

import pytest

from ocr_mini_bench.ocr.text_readers import (
    read_mistral_ocr_annotation_as_text,
    read_mistral_ocr_markdown,
    read_text_from_anthropic_content,
    read_text_from_gemini_response,
    read_text_from_mistral_chat_response,
    read_text_from_openai_response,
)


@pytest.mark.unit
def test_read_gemini_text_concatenates_parts() -> None:
    data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "  hello  "},
                        {"text": "world"},
                    ]
                }
            }
        ]
    }
    # Outer .trim() strips leading/trailing whitespace; inner part text is preserved.
    assert read_text_from_gemini_response(data) == "hello  \nworld"


@pytest.mark.unit
def test_read_gemini_text_empty_when_missing() -> None:
    assert read_text_from_gemini_response({}) == ""
    assert read_text_from_gemini_response({"candidates": []}) == ""
    assert read_text_from_gemini_response({"candidates": [{"content": {"parts": []}}]}) == ""


@pytest.mark.unit
def test_read_openai_text_prefers_output_text() -> None:
    assert read_text_from_openai_response({"output_text": "  result  "}) == "result"


@pytest.mark.unit
def test_read_openai_text_falls_back_to_output_array() -> None:
    data = {
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": "first"},
                    {"type": "reasoning", "text": "ignored"},
                ]
            },
            {"content": [{"type": "output_text", "text": "second"}]},
        ]
    }
    assert read_text_from_openai_response(data) == "first\nsecond"


@pytest.mark.unit
def test_read_anthropic_content_filters_to_text_blocks() -> None:
    # Plain-dict path (recorded fixtures).
    content = [
        {"type": "text", "text": "a"},
        {"type": "tool_use", "input": {}},
        {"type": "text", "text": "b"},
    ]
    assert read_text_from_anthropic_content(content) == "a\nb"


@pytest.mark.unit
def test_read_anthropic_content_supports_sdk_objects() -> None:
    class Block:
        def __init__(self, type_: str, text: str = "") -> None:
            self.type = type_
            self.text = text

    blocks = [Block("text", "x"), Block("tool_use"), Block("text", "y")]
    assert read_text_from_anthropic_content(list(blocks)) == "x\ny"


@pytest.mark.unit
def test_read_mistral_chat_text_handles_string_content() -> None:
    data = {"choices": [{"message": {"content": "  hello "}}]}
    assert read_text_from_mistral_chat_response(data) == "hello"


@pytest.mark.unit
def test_read_mistral_chat_text_handles_array_content() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "text", "text": "b"},
                    ]
                }
            }
        ]
    }
    assert read_text_from_mistral_chat_response(data) == "a\nb"


@pytest.mark.unit
def test_read_mistral_chat_text_missing_returns_empty() -> None:
    assert read_text_from_mistral_chat_response({}) == ""
    assert read_text_from_mistral_chat_response({"choices": []}) == ""


@pytest.mark.unit
def test_read_mistral_annotation_string_path_reparses_json() -> None:
    # When `document_annotation` is a JSON string, we reparse and re-serialize
    # compactly so the canonical raw text doesn't depend on the provider's
    # whitespace choices.
    data = {"document_annotation": '{"a": 1,    "b": 2}'}
    assert read_mistral_ocr_annotation_as_text(data) == '{"a":1,"b":2}'


@pytest.mark.unit
def test_read_mistral_annotation_record_path_serializes() -> None:
    data = {"document_annotation": {"a": 1}}
    assert read_mistral_ocr_annotation_as_text(data) == '{"a":1}'


@pytest.mark.unit
def test_read_mistral_annotation_missing_returns_empty() -> None:
    assert read_mistral_ocr_annotation_as_text({}) == ""


@pytest.mark.unit
def test_read_mistral_markdown_joins_pages() -> None:
    data = {"pages": [{"markdown": "one"}, {"markdown": ""}, {"markdown": "two"}]}
    assert read_mistral_ocr_markdown(data) == "one\n\ntwo"
