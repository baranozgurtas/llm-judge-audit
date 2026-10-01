"""Append-only, resumable 400-cell runner. One record per (pair_id, order); no retries."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_judge_audit.config import ORDERS
from llm_judge_audit.integrity import (
    CELL_SCHEMA_VERSION,
    IDENTITY_FIELDS,
    check_integrity,
    public_report,
    record_hash,
)
from llm_judge_audit.io_utils import canonical_json, read_json, sha256_text, write_json
from llm_judge_audit.ollama_client import ChatResult, JudgeClient
from llm_judge_audit.parser import parse_output
from llm_judge_audit.prompt import load_template, map_verdict, render_cell

CELLS_FILE = "cells.jsonl"
RUN_META_FILE = "run_meta.json"
SESSIONS_FILE = "sessions.jsonl"
PROGRESS_FILE = "progress.json"


class RunRefused(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def schedule(manifest: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Deterministic schedule: manifest order, original then swapped for each pair."""
    return [(m["pair_id"], order) for m in manifest for order in ORDERS]


def run_identity(run_id: str, freeze: dict[str, Any]) -> dict[str, Any]:
    ident = {"run_id": run_id}
    ident.update({f: freeze[f] for f in IDENTITY_FIELDS if f != "run_id"})
    return ident


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one complete line with a single write + fsync (O_APPEND, never rewrites)."""
    line = (canonical_json(record) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        written = os.write(fd, line)
        if written != len(line):
            raise OSError(f"short write: {written} of {len(line)} bytes")
        os.fsync(fd)
    finally:
        os.close(fd)


def build_record(
    identity: dict[str, Any],
    pair_id: str,
    order: str,
    mapping: dict[str, str],
    prompt: str,
    result: ChatResult,
    num_ctx: int,
) -> dict[str, Any]:
    parse: dict[str, Any] | None = None
    verdict: str | None = None
    mapped: str | None = None
    error = result.error
    if result.status == "timeout":
        status = "timeout"
    elif result.status != "ok" or result.content is None:
        status = "runtime_error"
    elif result.done_reason == "length":
        status, error = "invalid_output", "truncated_output: done_reason=length"
    elif (
        result.prompt_tokens is not None
        and result.prompt_tokens + (result.completion_tokens or 0) >= num_ctx
    ):
        status, error = "invalid_output", "context_limit_reached"
    else:
        parsed = parse_output(result.content)
        parse = parsed.as_dict()
        if parsed.ok and parsed.verdict is not None:
            status, verdict = "valid", parsed.verdict
            mapped = map_verdict(parsed.verdict, order)
        else:
            status, error = "invalid_output", parsed.error
    record: dict[str, Any] = {
        "schema_version": CELL_SCHEMA_VERSION,
        **identity,
        "pair_id": pair_id,
        "order": order,
        "display_mapping": mapping,
        "status": status,
        "raw_response": result.content,
        "parse": parse,
        "verdict_displayed": verdict,
        "mapped_verdict": mapped,
        "latency_s": round(result.latency_s, 4),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "done_reason": result.done_reason,
        "runtime": result.runtime,
        "error": error,
        "rendered_prompt_sha256": sha256_text(prompt),
        "timestamp_utc": utc_now(),
    }
    record["record_sha256"] = record_hash(record)
    return record


def run_benchmark(
    run_dir: Path,
    manifest: list[dict[str, Any]],
    freeze: dict[str, Any],
    live: dict[str, Any],
    client: JudgeClient,
    num_ctx: int,
    max_consecutive_connection_errors: int,
    run_meta_extra: dict[str, Any] | None = None,
    on_cell: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run (or resume) every missing cell. Returns the public integrity report."""
    for key in ("model_digest", "ollama_version"):
        if live.get(key) != freeze[key]:
            raise RunRefused(f"live {key} {live.get(key)!r} != frozen {freeze[key]!r}")
    run_id = run_dir.name
    identity = run_identity(run_id, freeze)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / RUN_META_FILE
    if meta_path.exists():
        meta = read_json(meta_path)
        diffs = [f for f in IDENTITY_FIELDS if meta["identity"].get(f) != identity.get(f)]
        if diffs or meta.get("freeze") != freeze:
            raise RunRefused(f"resume refused: run identity differs in {diffs or ['freeze']}")
    else:
        write_json(
            meta_path,
            {
                "identity": identity,
                "freeze": freeze,
                "created_utc": utc_now(),
                "expected_cells": 2 * len(manifest),
                **(run_meta_extra or {}),
            },
        )
    cells_path = run_dir / CELLS_FILE
    pair_ids = [m["pair_id"] for m in manifest]
    before = check_integrity(cells_path, pair_ids, identity)
    if not before["integrity_ok"]:
        raise RunRefused(f"resume refused: integrity problems {public_report(before)['counts']}")
    done = set(before["accepted"])
    by_id = {m["pair_id"]: m for m in manifest}
    template = load_template()
    session = {
        "session_start_utc": utc_now(),
        "cells_already_recorded": len(done),
        **(run_meta_extra or {}),
    }
    attempted = 0
    consecutive_conn_errors = 0
    aborted: str | None = None
    todo = [cell for cell in schedule(manifest) if cell not in done]
    try:
        for i, (pid, order) in enumerate(todo, start=1):
            rendered = render_cell(by_id[pid], order, template)
            result = client.chat(rendered.prompt)
            record = build_record(
                identity, pid, order, rendered.mapping, rendered.prompt, result, num_ctx
            )
            append_record(cells_path, record)
            attempted += 1
            write_json(
                run_dir / PROGRESS_FILE,
                {
                    "run_id": run_id,
                    "recorded": len(done) + attempted,
                    "expected": 2 * len(manifest),
                    "last_cell": [pid, order],
                    "updated_utc": utc_now(),
                },
            )
            if on_cell:
                on_cell(i, len(todo), record)
            consecutive_conn_errors = consecutive_conn_errors + 1 if result.connection_error else 0
            if consecutive_conn_errors >= max_consecutive_connection_errors:
                aborted = f"stopped after {consecutive_conn_errors} consecutive connection errors"
                break
    except KeyboardInterrupt:
        aborted = "interrupted by user"
    finally:
        session.update(
            {"session_end_utc": utc_now(), "cells_attempted": attempted, "aborted": aborted}
        )
        append_record(run_dir / SESSIONS_FILE, session)
    report = public_report(check_integrity(cells_path, pair_ids, identity))
    report["aborted"] = aborted
    return report
