"""Frozen judge prompt (v1) and order-specific rendering.

`render_prompt` accepts only the question text and the two displayed response texts, so human
labels, annotator ids, source model names and categories cannot reach the judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from llm_judge_audit.config import PROJECT_ROOT
from llm_judge_audit.io_utils import sha256_text

PROMPT_VERSION = "judge-prompt-v1"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "judge_prompt_v1.txt"
_PLACEHOLDER_RE = re.compile(r"\{\{(QUESTION|RESPONSE_A|RESPONSE_B)\}\}")

# Displayed position -> canonical candidate for each order.
DISPLAY_MAPPING: dict[str, dict[str, str]] = {
    "original": {"A": "cand_1", "B": "cand_2"},
    "swapped": {"A": "cand_2", "B": "cand_1"},
}


def load_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def prompt_hash() -> str:
    """SHA-256 of the exact template bytes."""
    return sha256_text(load_template())


def render_prompt(
    question: str, response_a: str, response_b: str, template: str | None = None
) -> str:
    """Single-pass substitution, so placeholder-like text inside answers is left verbatim."""
    tpl = load_template() if template is None else template
    values = {"QUESTION": question, "RESPONSE_A": response_a, "RESPONSE_B": response_b}
    found = _PLACEHOLDER_RE.findall(tpl)
    if sorted(found) != sorted(values):
        raise ValueError(f"template must contain each placeholder exactly once, found {found}")
    return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], tpl)


@dataclass(frozen=True)
class RenderedCell:
    pair_id: str
    order: str
    mapping: dict[str, str]
    prompt: str


def render_cell(pair: dict[str, Any], order: str, template: str | None = None) -> RenderedCell:
    """Render one (pair, order) cell from a manifest record, reading only content fields."""
    mapping = DISPLAY_MAPPING[order]
    cands = pair["candidates"]
    prompt = render_prompt(
        pair["question"],
        cands[mapping["A"]]["text"],
        cands[mapping["B"]]["text"],
        template,
    )
    return RenderedCell(pair["pair_id"], order, dict(mapping), prompt)


def map_verdict(verdict: str, order: str) -> str:
    """Translate a displayed verdict (A/B/tie) back to the canonical candidate identity."""
    if verdict == "tie":
        return "tie"
    return DISPLAY_MAPPING[order][verdict]
