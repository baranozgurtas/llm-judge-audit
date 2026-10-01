"""Integrity validation of an append-only cell log against the manifest and the freeze."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm_judge_audit.config import ORDERS
from llm_judge_audit.io_utils import canonical_json, sha256_text
from llm_judge_audit.prompt import DISPLAY_MAPPING, map_verdict

CELL_SCHEMA_VERSION = "cell-v1"
STATUSES = ("valid", "invalid_output", "runtime_error", "timeout")
# Fields in every cell that must equal the run's frozen identity.
IDENTITY_FIELDS = (
    "run_id",
    "study_hash",
    "config_sha256",
    "prompt_sha256",
    "parser_version",
    "parser_sha256",
    "manifest_sha256",
    "model_tag",
    "model_digest",
    "ollama_version",
)
REQUIRED_FIELDS = (
    "schema_version",
    *IDENTITY_FIELDS,
    "pair_id",
    "order",
    "display_mapping",
    "status",
    "raw_response",
    "parse",
    "verdict_displayed",
    "mapped_verdict",
    "latency_s",
    "prompt_tokens",
    "completion_tokens",
    "done_reason",
    "runtime",
    "error",
    "rendered_prompt_sha256",
    "timestamp_utc",
    "record_sha256",
)


def record_hash(record: dict[str, Any]) -> str:
    return sha256_text(canonical_json({k: v for k, v in record.items() if k != "record_sha256"}))


def load_cells(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    """Parse the cell log. Returns ([(line_no, record)], [corruption issues])."""
    records: list[tuple[int, dict[str, Any]]] = []
    issues: list[dict[str, Any]] = []
    if not path.exists():
        return records, issues
    data = path.read_bytes()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    elif lines:
        issues.append({"line": len(lines), "reason": "torn_tail_without_newline"})
    for i, raw in enumerate(lines, start=1):
        try:
            rec = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            issues.append({"line": i, "reason": "invalid_json"})
            continue
        if not isinstance(rec, dict):
            issues.append({"line": i, "reason": "not_an_object"})
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            issues.append({"line": i, "reason": f"missing_fields:{missing}"})
            continue
        if rec["record_sha256"] != record_hash(rec):
            issues.append({"line": i, "reason": "record_hash_mismatch"})
            continue
        records.append((i, rec))
    return records, issues


def _semantic_problem(rec: dict[str, Any]) -> str | None:
    status, order = rec["status"], rec["order"]
    if rec["schema_version"] != CELL_SCHEMA_VERSION:
        return "schema_version"
    if status not in STATUSES:
        return "unknown_status"
    if rec["display_mapping"] != DISPLAY_MAPPING.get(order):
        return "display_mapping"
    if status == "valid":
        parse = rec["parse"]
        if not parse or not parse.get("ok") or parse.get("verdict") != rec["verdict_displayed"]:
            return "valid_without_parse"
        if rec["mapped_verdict"] != map_verdict(rec["verdict_displayed"], order):
            return "mapped_verdict"
    elif rec["mapped_verdict"] is not None or rec["verdict_displayed"] is not None:
        return "verdict_on_failed_cell"
    return None


def check_integrity(
    cells_path: Path, manifest_ids: list[str], identity: dict[str, Any]
) -> dict[str, Any]:
    """Classify every line; return counts, issue lists, and the completion state."""
    records, corrupt = load_cells(cells_path)
    expected = [(pid, order) for pid in manifest_ids for order in ORDERS]
    expected_set = set(expected)
    seen: dict[tuple[str, str], int] = {}
    duplicates, mismatched, out_of_manifest, inconsistent = [], [], [], []
    accepted: dict[tuple[str, str], dict[str, Any]] = {}
    for line, rec in records:
        key = (rec["pair_id"], rec["order"])
        if key not in expected_set:
            out_of_manifest.append({"line": line, "pair_id": key[0], "order": key[1]})
            continue
        diffs = [f for f in IDENTITY_FIELDS if rec.get(f) != identity.get(f)]
        if diffs:
            mismatched.append({"line": line, "pair_id": key[0], "order": key[1], "fields": diffs})
            continue
        problem = _semantic_problem(rec)
        if problem:
            inconsistent.append(
                {"line": line, "pair_id": key[0], "order": key[1], "reason": problem}
            )
            continue
        if key in seen:
            duplicates.append(
                {"line": line, "first_line": seen[key], "pair_id": key[0], "order": key[1]}
            )
            continue
        seen[key] = line
        accepted[key] = rec
    status_counts = Counter(r["status"] for r in accepted.values())
    missing = [{"pair_id": p, "order": o} for p, o in expected if (p, o) not in accepted]
    integrity_ok = not (corrupt or duplicates or mismatched or out_of_manifest or inconsistent)
    counts = {
        "expected": len(expected),
        "attempted": len(accepted),
        "valid": status_counts.get("valid", 0),
        "invalid_output": status_counts.get("invalid_output", 0),
        "runtime_error": status_counts.get("runtime_error", 0),
        "timeout": status_counts.get("timeout", 0),
        "missing": len(missing),
        "complete": status_counts.get("valid", 0),
        "log_lines": len(records) + len(corrupt),
    }
    if not integrity_ok:
        state = "INTEGRITY_FAILED"
    elif counts["attempted"] == 0:
        state = "NOT_STARTED"
    elif counts["valid"] == counts["expected"]:
        state = "COMPLETE"
    else:
        state = "PARTIAL"
    return {
        "completion_state": state,
        "integrity_ok": integrity_ok,
        "counts": counts,
        "corrupt": corrupt,
        "duplicates": duplicates,
        "mismatched": mismatched,
        "out_of_manifest": out_of_manifest,
        "inconsistent": inconsistent,
        "missing_cells": missing,
        "accepted": accepted,
    }


def public_report(report: dict[str, Any]) -> dict[str, Any]:
    """The integrity report without the in-memory accepted records."""
    return {k: v for k, v in report.items() if k != "accepted"}
