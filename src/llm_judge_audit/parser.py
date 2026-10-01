"""Strict, model-independent validation of judge output (parser-v1).

Output contract: the visible response, after stripping surrounding whitespace, is exactly one
JSON object with exactly the keys "verdict" and "rationale"; no duplicate keys; verdict is the
string "A", "B" or "tie"; rationale is null or a non-empty string of at most 300 characters and
at most two sentences (a sentence ends at '.', '!' or '?' followed by whitespace or end of text).
Anything else is an invalid output, never a verdict. No repair, no retries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

PARSER_VERSION = "parser-v1"
VERDICTS = ("A", "B", "tie")
RATIONALE_MAX_CHARS = 300
RATIONALE_MAX_SENTENCES = 2
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# JSON schema passed to Ollama's constrained decoding (`format`). Validation below is independent.
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "rationale": {"type": ["string", "null"], "maxLength": RATIONALE_MAX_CHARS},
    },
    "required": ["verdict", "rationale"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    verdict: str | None
    rationale: str | None
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "verdict": self.verdict,
            "rationale": self.rationale,
            "error": self.error,
        }


def _fail(error: str) -> ParseResult:
    return ParseResult(False, None, None, error)


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate keys")
    return dict(pairs)


def count_sentences(text: str) -> int:
    return len([s for s in _SENTENCE_SPLIT.split(text.strip()) if s])


def parse_output(raw: str) -> ParseResult:
    text = raw.strip()
    if not text:
        return _fail("empty_output")
    try:
        obj = json.loads(text, object_pairs_hook=_no_duplicates)
    except ValueError as exc:
        return _fail(f"invalid_json: {exc}")
    if not isinstance(obj, dict):
        return _fail("not_an_object")
    if set(obj) != {"verdict", "rationale"}:
        return _fail(f"wrong_keys: {sorted(obj)}")
    verdict, rationale = obj["verdict"], obj["rationale"]
    if not isinstance(verdict, str) or verdict not in VERDICTS:
        return _fail(f"bad_verdict: {verdict!r}")
    if rationale is not None:
        if not isinstance(rationale, str) or not rationale.strip():
            return _fail("bad_rationale_type_or_empty")
        if len(rationale) > RATIONALE_MAX_CHARS:
            return _fail("rationale_too_long")
        if count_sentences(rationale) > RATIONALE_MAX_SENTENCES:
            return _fail("rationale_too_many_sentences")
    return ParseResult(True, verdict, rationale, None)
