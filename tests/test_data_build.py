"""Data construction tests on SAMPLE DATA (synthetic) fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_judge_audit.data.build import SamplingError, build_artifacts, stratified_sample
from llm_judge_audit.data.rebuild import RebuildMismatch, check, rebuild
from llm_judge_audit.data.schema import SchemaError
from llm_judge_audit.io_utils import read_jsonl
from synthetic import default_rows, row, write_source


def _build(tmp_path: Path, rows: list[dict[str, Any]] | None = None) -> dict[str, bytes]:
    cfg = write_source(tmp_path, rows)
    artifacts, _ = build_artifacts(cfg, tmp_path)
    return artifacts


def _pairs(artifacts: dict[str, bytes]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in artifacts["pairs.jsonl"].decode().splitlines()]


def _by_q_models(pairs: list[dict[str, Any]], qid: int, models: set[str]) -> dict[str, Any]:
    for p in pairs:
        got = {m for c in p["candidates"].values() for m in c["source_models"]}
        if p["question_id"] == qid and got == models:
            return p
    raise KeyError((qid, models))


def test_flipped_rows_merge_into_one_pair_and_votes_map_to_content(tmp_path: Path) -> None:
    pairs = _pairs(_build(tmp_path))
    p = _by_q_models(pairs, 1, {"mx", "my"})
    x_cand = next(k for k, c in p["candidates"].items() if c["source_models"] == ["mx"])
    assert p["human"]["counts"][x_cand] == 2
    assert p["human"]["aggregate"] == x_cand
    assert p["human"]["n_votes_retained"] == 2


def test_tie_stays_tie_and_split_is_unresolved(tmp_path: Path) -> None:
    pairs = _pairs(_build(tmp_path))
    assert _by_q_models(pairs, 1, {"mx", "mz"})["human"]["aggregate"] == "tie"
    assert _by_q_models(pairs, 4, {"mx", "mz"})["human"]["aggregate"] == "unresolved"


def test_conflicting_votes_dropped_and_logged(tmp_path: Path) -> None:
    artifacts = _build(tmp_path)
    p = _by_q_models(_pairs(artifacts), 2, {"mx", "mz"})
    assert p["human"]["n_votes_source"] == 3
    assert p["human"]["n_votes_retained"] == 1
    assert p["human"]["aggregate"] == "tie"
    audit = json.loads(artifacts["audit.json"])
    conflicts = [
        a for a in audit["vote_adjustments"] if a["reason"] == "conflicting_votes_same_annotator"
    ]
    assert {a["annotator"] for a in conflicts} == {"expert_9"}
    assert len(conflicts) == 2 and all(a["action"] == "dropped" for a in conflicts)


def test_duplicate_votes_counted_once_and_logged(tmp_path: Path) -> None:
    artifacts = _build(tmp_path)
    p = _by_q_models(_pairs(artifacts), 3, {"mx", "mz"})
    assert p["human"]["counts"]["tie"] == 1
    audit = json.loads(artifacts["audit.json"])
    dups = [a for a in audit["vote_adjustments"] if a["reason"] == "duplicate_vote_same_annotator"]
    assert sorted(a["action"] for a in dups) == ["dropped", "kept"]


def test_identical_texts_and_turn2_excluded_with_reasons(tmp_path: Path) -> None:
    artifacts = _build(tmp_path)
    audit = json.loads(artifacts["audit.json"])
    reasons = {e["reason"] for e in audit["row_exclusions"]}
    assert reasons == {"identical_candidate_texts", "turn_2_out_of_scope"}
    assert [e["reason"] for e in audit["pair_exclusions"]] == ["identical_candidate_texts"]
    for p in _pairs(artifacts):
        assert p["candidates"]["cand_1"]["text"] != p["candidates"]["cand_2"]["text"]
        assert p["turn"] == 1
    totals = audit["totals"]
    n_rows = len(default_rows())
    assert totals["source_rows"] == n_rows
    excluded = sum(totals["row_exclusions_by_reason"].values())
    in_pairs = sum(len(p["source_rows"]) for p in _pairs(artifacts))
    assert excluded + in_pairs == n_rows


def test_manifest_is_stratified_and_deterministic(tmp_path: Path) -> None:
    a1 = _build(tmp_path / "one")
    a2 = _build(tmp_path / "two")
    assert a1 == a2
    manifest = [json.loads(x) for x in a1["manifest.jsonl"].decode().splitlines()]
    assert len(manifest) == 4
    assert sorted(m["category"] for m in manifest) == ["math", "math", "writing", "writing"]
    assert len({m["pair_id"] for m in manifest}) == 4


def test_sampling_stops_when_stratum_too_small(tmp_path: Path) -> None:
    pairs = _pairs(_build(tmp_path))
    with pytest.raises(SamplingError, match="eligible counts below"):
        stratified_sample(pairs, ["writing", "math"], 40, seed=1)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("winner", "tie (bothbad)", "unknown winner"),
        ("turn", 3, "unknown turn"),
        ("judge", "gpt4_pair", "not a human annotator"),
        ("question_id", 999, "not in MT-bench"),
    ],
)
def test_schema_violations_fail_loudly(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    rows = default_rows()
    rows[0][field] = value
    with pytest.raises(SchemaError, match=match):
        _build(tmp_path, rows)


def test_user_turn_mismatch_fails(tmp_path: Path) -> None:
    rows = default_rows()
    rows[0]["conversation_a"][0]["content"] = "different question"
    with pytest.raises(SchemaError, match="user turns differ"):
        _build(tmp_path, rows)


def test_rebuild_writes_identical_bytes_and_check_detects_tampering(tmp_path: Path) -> None:
    cfg = write_source(tmp_path, [*default_rows(), row(7, "ma", "mb", "p", "q", "tie", "expert_4")])
    out = tmp_path / "derived"
    h1 = rebuild(cfg, out, tmp_path)
    h2 = rebuild(cfg, out, tmp_path)
    assert h1 == h2
    check(cfg, out, tmp_path)
    manifest = read_jsonl(out / "manifest.jsonl")
    assert manifest
    (out / "manifest.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RebuildMismatch, match=r"manifest\.jsonl"):
        check(cfg, out, tmp_path)
