"""Loud validation of the real source schema and semantics.

Observed at revision f7d2896 (human split): 3,355 rows, columns below, each conversation is
exactly [user, assistant, user, assistant], `turn` in {1, 2} names the judged turn, `winner`
in {model_a, model_b, tie}, `judge` is a pseudonymous annotator id (expert_N / author_N).
Anything else raises SchemaError instead of being guessed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

CONV_TYPE = "list<item: struct<content: string, role: string>>"
EXPECTED_COLUMNS: dict[str, str] = {
    "question_id": "int64",
    "model_a": "string",
    "model_b": "string",
    "winner": "string",
    "judge": "string",
    "conversation_a": CONV_TYPE,
    "conversation_b": CONV_TYPE,
    "turn": "int64",
}
WINNER_VALUES = frozenset({"model_a", "model_b", "tie"})
TURN_VALUES = frozenset({1, 2})
ROLES = ("user", "assistant", "user", "assistant")
ANNOTATOR_RE = re.compile(r"^(expert|author)_\d+$")
QUESTION_KEYS = frozenset({"question_id", "category", "turns"})


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class SourceRow:
    source_row: int
    question_id: int
    model_a: str
    model_b: str
    winner: str
    annotator: str
    turn: int
    question_turn1: str
    text_a: str
    text_b: str


@dataclass(frozen=True)
class Question:
    question_id: int
    category: str
    turns: tuple[str, ...]


def load_questions(path: Path) -> dict[int, Question]:
    out: dict[int, Question] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if set(obj) - {"reference"} != QUESTION_KEYS:
            raise SchemaError(f"question.jsonl line {lineno}: unexpected keys {sorted(obj)}")
        qid, cat, turns = obj["question_id"], obj["category"], obj["turns"]
        if not isinstance(qid, int) or not isinstance(cat, str) or not isinstance(turns, list):
            raise SchemaError(f"question.jsonl line {lineno}: bad field types")
        if len(turns) != 2 or not all(isinstance(t, str) and t for t in turns):
            raise SchemaError(f"question.jsonl line {lineno}: expected two non-empty turns")
        if qid in out:
            raise SchemaError(f"question.jsonl: duplicate question_id {qid}")
        out[qid] = Question(qid, cat, tuple(turns))
    return out


def validate_rows(raw: list[dict[str, Any]], questions: dict[int, Question]) -> list[SourceRow]:
    """Validate every raw row and return typed rows. Raises on the first violation."""
    rows: list[SourceRow] = []
    for idx, r in enumerate(raw):
        where = f"source row {idx}"
        if set(r) != set(EXPECTED_COLUMNS):
            raise SchemaError(f"{where}: columns {sorted(r)} != {sorted(EXPECTED_COLUMNS)}")
        qid = r["question_id"]
        if not isinstance(qid, int) or qid not in questions:
            raise SchemaError(f"{where}: question_id {qid!r} not in MT-bench questions")
        if r["winner"] not in WINNER_VALUES:
            raise SchemaError(f"{where}: unknown winner value {r['winner']!r}")
        if r["turn"] not in TURN_VALUES:
            raise SchemaError(f"{where}: unknown turn value {r['turn']!r}")
        annotator = r["judge"]
        if not isinstance(annotator, str) or not ANNOTATOR_RE.match(annotator):
            raise SchemaError(f"{where}: judge {annotator!r} is not a human annotator id")
        model_a, model_b = r["model_a"], r["model_b"]
        if not model_a or not model_b or model_a == model_b:
            raise SchemaError(f"{where}: invalid model pair {model_a!r} vs {model_b!r}")
        q = questions[qid]
        convs = (r["conversation_a"], r["conversation_b"])
        for side, conv in zip("ab", convs, strict=True):
            if not isinstance(conv, list) or len(conv) != 4:
                raise SchemaError(f"{where}: conversation_{side} must have 4 messages")
            if tuple(m.get("role") for m in conv) != ROLES:
                raise SchemaError(f"{where}: conversation_{side} roles are not {ROLES}")
            if any(not isinstance(m.get("content"), str) for m in conv):
                raise SchemaError(f"{where}: conversation_{side} has non-string content")
            if conv[0]["content"] != q.turns[0] or conv[2]["content"] != q.turns[1]:
                raise SchemaError(f"{where}: conversation_{side} user turns differ from question")
        rows.append(
            SourceRow(
                source_row=idx,
                question_id=qid,
                model_a=model_a,
                model_b=model_b,
                winner=r["winner"],
                annotator=annotator,
                turn=r["turn"],
                question_turn1=q.turns[0],
                text_a=convs[0][1]["content"],
                text_b=convs[1][1]["content"],
            )
        )
    return rows


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    actual = {f.name: str(f.type) for f in table.schema}
    if actual != EXPECTED_COLUMNS:
        raise SchemaError(f"parquet schema mismatch: {actual}")
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows
