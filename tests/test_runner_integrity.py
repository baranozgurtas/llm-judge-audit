"""Runner, resume, no-retry, and integrity behaviour with a fake judge (SAMPLE DATA)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_judge_audit.integrity import check_integrity, record_hash
from llm_judge_audit.io_utils import read_jsonl
from llm_judge_audit.ollama_client import ChatResult
from llm_judge_audit.runner import CELLS_FILE, RunRefused, run_benchmark, schedule
from llm_judge_audit.sample_data import FakeJudge, sample_freeze, synthetic_manifest

MANIFEST = synthetic_manifest(10)
FREEZE = sample_freeze("manifest-hash")
LIVE = {"model_digest": FREEZE["model_digest"], "ollama_version": FREEZE["ollama_version"]}


def _run(run_dir: Path, client: FakeJudge, **kw: Any) -> dict[str, Any]:
    return run_benchmark(
        run_dir,
        MANIFEST,
        kw.get("freeze", FREEZE),
        kw.get("live", LIVE),
        client,
        num_ctx=8192,
        max_consecutive_connection_errors=3,
    )


def _ids() -> list[str]:
    return [m["pair_id"] for m in MANIFEST]


def _identity(run_dir: Path) -> dict[str, Any]:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    out: dict[str, Any] = meta["identity"]
    return out


def test_schedule_has_two_orders_per_pair() -> None:
    cells = schedule(MANIFEST)
    assert len(cells) == 2 * len(MANIFEST) == len(set(cells))


def test_full_run_records_every_cell_once(tmp_path: Path) -> None:
    client = FakeJudge()
    report = _run(tmp_path / "r", client)
    assert len(client.calls) == 20
    assert report["counts"]["attempted"] == 20
    assert report["integrity_ok"]
    # FakeJudge makes call index 10 invalid -> PARTIAL, never COMPLETE
    assert report["counts"]["invalid_output"] == 1
    assert report["completion_state"] == "PARTIAL"


def test_failures_are_terminal_no_retry_and_resume_skips(tmp_path: Path) -> None:
    run_dir = tmp_path / "r"
    script: dict[int, ChatResult | str] = {
        0: ChatResult("timeout", None, 300.0, error="timeout"),
        1: ChatResult("runtime_error", None, 0.1, error="HTTP 500"),
        2: '{"verdict": "A", "rationale": null}',
    }
    first = FakeJudge(script)
    _run(run_dir, first)
    before = (run_dir / CELLS_FILE).read_bytes()
    second = FakeJudge()
    report = _run(run_dir, second)
    assert second.calls == []  # nothing re-attempted, failures stay terminal
    assert (run_dir / CELLS_FILE).read_bytes() == before
    assert report["counts"]["timeout"] == 1 and report["counts"]["runtime_error"] == 1
    assert report["completion_state"] == "PARTIAL"


def test_resume_after_interrupt_continues_without_overwrite(tmp_path: Path) -> None:
    run_dir = tmp_path / "r"

    class Interrupting(FakeJudge):
        def chat(self, prompt: str) -> ChatResult:
            if len(self.calls) == 5:
                raise KeyboardInterrupt
            return super().chat(prompt)

    report = _run(run_dir, Interrupting())
    assert report["aborted"] == "interrupted by user"
    assert report["counts"]["attempted"] == 5
    prefix = (run_dir / CELLS_FILE).read_bytes()
    client = FakeJudge()
    report = _run(run_dir, client)
    assert len(client.calls) == 15
    assert (run_dir / CELLS_FILE).read_bytes().startswith(prefix)
    assert report["counts"]["attempted"] == 20
    assert len(read_jsonl(run_dir / "sessions.jsonl")) == 2


def test_connection_errors_stop_the_run(tmp_path: Path) -> None:
    down = ChatResult("runtime_error", None, 0.0, error="refused", connection_error=True)
    report = _run(tmp_path / "r", FakeJudge(dict.fromkeys(range(20), down)))
    assert report["counts"]["attempted"] == 3
    assert report["counts"]["missing"] == 17
    assert "connection errors" in report["aborted"]


def test_truncation_and_invalid_json_are_not_verdicts(tmp_path: Path) -> None:
    script: dict[int, ChatResult | str] = {
        0: ChatResult("ok", '{"verdict": "A", "rationale": "cut', 1.0, "length", 100, 256),
        1: "```json {}```",
    }
    _run(tmp_path / "r", FakeJudge(script))
    recs = read_jsonl(tmp_path / "r" / CELLS_FILE)
    assert recs[0]["status"] == "invalid_output" and recs[0]["mapped_verdict"] is None
    assert recs[0]["error"].startswith("truncated_output")
    assert recs[1]["status"] == "invalid_output" and recs[1]["verdict_displayed"] is None


def test_resume_refused_on_identity_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "r"
    _run(run_dir, FakeJudge({}))
    changed = {**FREEZE, "prompt_sha256": "different"}
    with pytest.raises(RunRefused, match="resume refused"):
        _run(run_dir, FakeJudge(), freeze=changed)
    with pytest.raises(RunRefused, match="live model_digest"):
        _run(run_dir, FakeJudge(), live={**LIVE, "model_digest": "other"})


def _append_raw(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def test_integrity_detects_every_problem_class(tmp_path: Path) -> None:
    run_dir = tmp_path / "r"
    _run(run_dir, FakeJudge())
    cells = run_dir / CELLS_FILE
    recs = read_jsonl(cells)
    ident = _identity(run_dir)

    dup = dict(recs[0])
    _append_raw(cells, json.dumps(dup) + "\n")
    out = dict(recs[1], pair_id="not_in_manifest")
    out["record_sha256"] = record_hash(out)
    _append_raw(cells, json.dumps(out) + "\n")
    mism = dict(recs[2], model_digest="other")
    mism["record_sha256"] = record_hash(mism)
    _append_raw(cells, json.dumps(mism) + "\n")
    tampered = dict(
        recs[3], mapped_verdict="cand_2" if recs[3]["mapped_verdict"] != "cand_2" else "cand_1"
    )
    _append_raw(cells, json.dumps(tampered) + "\n")
    bad_map = dict(
        recs[4], mapped_verdict="tie" if recs[4]["mapped_verdict"] != "tie" else "cand_1"
    )
    bad_map["record_sha256"] = record_hash(bad_map)
    _append_raw(cells, json.dumps(bad_map) + "\n")
    _append_raw(cells, "{not json\n")
    _append_raw(cells, '{"torn": ')

    rep = check_integrity(cells, _ids(), ident)
    assert not rep["integrity_ok"]
    assert rep["completion_state"] == "INTEGRITY_FAILED"
    assert len(rep["duplicates"]) == 1
    assert len(rep["out_of_manifest"]) == 1
    assert len(rep["mismatched"]) == 1 and rep["mismatched"][0]["fields"] == ["model_digest"]
    reasons = sorted(c["reason"] for c in rep["corrupt"])
    assert reasons == [
        "invalid_json",
        "invalid_json",
        "record_hash_mismatch",
        "torn_tail_without_newline",
    ]
    assert len(rep["inconsistent"]) == 1
    with pytest.raises(RunRefused, match="integrity"):
        _run(run_dir, FakeJudge())


def test_missing_cells_reported(tmp_path: Path) -> None:
    run_dir = tmp_path / "r"
    _run(run_dir, FakeJudge())
    cells = run_dir / CELLS_FILE
    lines = cells.read_text().splitlines(keepends=True)
    cells.write_text("".join(lines[:-2]))
    rep = check_integrity(cells, _ids(), _identity(run_dir))
    assert rep["integrity_ok"] and rep["counts"]["missing"] == 2
    assert rep["completion_state"] == "PARTIAL"


def test_complete_only_when_all_cells_valid(tmp_path: Path) -> None:
    script: dict[int, ChatResult | str] = {10: '{"verdict": "B", "rationale": null}'}
    report = _run(tmp_path / "r", FakeJudge(script))
    assert report["counts"]["valid"] == 20
    assert report["completion_state"] == "COMPLETE"
