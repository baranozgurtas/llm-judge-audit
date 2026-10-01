"""SAMPLE DATA: synthetic fixtures for offline tests. Never real source data or model output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SAMPLE_LABEL = "SAMPLE DATA"
CATEGORIES = ["writing", "math"]


def question(qid: int) -> dict[str, Any]:
    cat = CATEGORIES[qid % 2]
    return {
        "question_id": qid,
        "category": cat,
        "turns": [f"SAMPLE DATA question {qid}?", f"SAMPLE DATA follow-up {qid}?"],
    }


def row(
    qid: int,
    model_a: str,
    model_b: str,
    text_a: str,
    text_b: str,
    winner: str,
    judge: str,
    turn: int = 1,
) -> dict[str, Any]:
    q = question(qid)

    def conv(text: str) -> list[dict[str, str]]:
        return [
            {"content": q["turns"][0], "role": "user"},
            {"content": text, "role": "assistant"},
            {"content": q["turns"][1], "role": "user"},
            {"content": f"{text} (turn 2)", "role": "assistant"},
        ]

    return {
        "question_id": qid,
        "model_a": model_a,
        "model_b": model_b,
        "winner": winner,
        "judge": judge,
        "conversation_a": conv(text_a),
        "conversation_b": conv(text_b),
        "turn": turn,
    }


def default_rows() -> list[dict[str, Any]]:
    """Rows covering ties, unresolved, duplicates, conflicts, identical texts, turn 2."""
    rows: list[dict[str, Any]] = []
    for qid in range(1, 9):
        a, b = f"SAMPLE answer x for {qid} short", f"SAMPLE answer y for {qid} which is longer"
        rows.append(row(qid, "mx", "my", a, b, "model_a", "expert_1"))
        rows.append(row(qid, "my", "mx", b, a, "model_b", "expert_2"))  # same pair, flipped
        rows.append(row(qid, "mx", "mz", a, f"SAMPLE z {qid}", "tie", "expert_1"))
    # q1 (mx,mz): add a tie-breaking vote -> stays tie with 2 tie votes
    rows.append(row(1, "mz", "mx", "SAMPLE z 1", "SAMPLE answer x for 1 short", "tie", "author_0"))
    # q2 (mx,mz): conflicting votes by expert_9 are dropped entirely
    rows.append(
        row(2, "mx", "mz", "SAMPLE answer x for 2 short", "SAMPLE z 2", "model_a", "expert_9")
    )
    rows.append(
        row(2, "mx", "mz", "SAMPLE answer x for 2 short", "SAMPLE z 2", "model_b", "expert_9")
    )
    # q3 (mx,mz): duplicate identical votes by expert_1 (already has tie) -> one kept
    rows.append(row(3, "mz", "mx", "SAMPLE z 3", "SAMPLE answer x for 3 short", "tie", "expert_1"))
    # q4 (mx,mz): split vote 1 cand vs 1 tie -> unresolved
    rows.append(
        row(4, "mz", "mx", "SAMPLE z 4", "SAMPLE answer x for 4 short", "model_a", "expert_3")
    )
    # identical texts -> excluded
    rows.append(row(5, "mx", "mq", "SAME TEXT", "SAME TEXT", "tie", "expert_1"))
    # turn 2 -> excluded
    rows.append(row(6, "mx", "my", "t2 a", "t2 b", "model_a", "expert_1", turn=2))
    return rows


def write_source(root: Path, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Write synthetic raw files under `root` and return a matching config."""
    rows = default_rows() if rows is None else rows
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    conv_t = pa.list_(
        pa.field("item", pa.struct([("content", pa.string()), ("role", pa.string())]))
    )
    schema = pa.schema(
        [
            ("question_id", pa.int64()),
            ("model_a", pa.string()),
            ("model_b", pa.string()),
            ("winner", pa.string()),
            ("judge", pa.string()),
            ("conversation_a", conv_t),
            ("conversation_b", conv_t),
            ("turn", pa.int64()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        raw / "human.parquet",
        use_compliant_nested_type=False,
    )
    (raw / "question.jsonl").write_text(
        "".join(json.dumps(question(q)) + "\n" for q in range(1, 9)), encoding="utf-8"
    )
    from llm_judge_audit.io_utils import sha256_file

    return {
        "study_name": "SAMPLE DATA study",
        "source": {
            "dataset": {
                "hf_repo": "sample/sample",
                "hf_revision": "0" * 40,
                "split": "human",
                "path": "human.parquet",
                "url": "file://sample",
                "sha256": sha256_file(raw / "human.parquet"),
                "local_file": "data/raw/human.parquet",
                "license": "SAMPLE",
            },
            "questions": {
                "repo": "sample/q",
                "revision": "0" * 40,
                "path": "question.jsonl",
                "url": "file://sample",
                "sha256": sha256_file(raw / "question.jsonl"),
                "local_file": "data/raw/question.jsonl",
                "license": "SAMPLE",
            },
        },
        "sampling": {
            "rule_version": "sampling-v1",
            "seed": 1,
            "n_pairs": 4,
            "turn": 1,
            "allocation": "equal",
            "categories": CATEGORIES,
        },
        "aggregation": {"rule_version": "aggregation-v1"},
    }
