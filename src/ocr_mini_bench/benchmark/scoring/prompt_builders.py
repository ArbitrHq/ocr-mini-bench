"""Build per-document system/user prompts from the prompt templates and
ground-truth key list. Mirrors `src/benchmark/scoring/prompt-builders.ts`.
"""

from __future__ import annotations

from ..types import GroundTruthDocument


def _build_key_instruction(document: GroundTruthDocument) -> str:
    lines = []
    for key in document.keys:
        data_type = key.data_type if key.data_type is not None else "unknown"
        critical = "yes" if key.critical else "no"
        lines.append(f"- {key.name} (type: {data_type}, critical: {critical})")
    return "\n".join(lines)


def _inject_requested_keys(template: str, key_instruction: str) -> str:
    if "{{REQUESTED_KEYS}}" in template:
        return template.replace("{{REQUESTED_KEYS}}", key_instruction)
    return f"{template.strip()}\n\nRequested keys:\n{key_instruction}"


def build_benchmark_system_prompt(template: str) -> str:
    return template.replace("{{REQUESTED_KEYS}}", "").strip()


def build_benchmark_user_prompt(template: str, document: GroundTruthDocument) -> str:
    return _inject_requested_keys(template, _build_key_instruction(document))
