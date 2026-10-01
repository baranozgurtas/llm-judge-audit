"""Integrity tests on the committed real derived artifacts (no network, no model)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from llm_judge_audit.config import DATA_DERIVED, PROJECT_ROOT, load_config
from llm_judge_audit.data.build import Vote, aggregate_votes
from llm_judge_audit.data.download import verify_raw
from llm_judge_audit.data.rebuild import check
from llm_judge_audit.io_utils import read_json, read_jsonl, sha256_file

CFG = load_config()


@pytest.fixture(scope="module")
def manifest() -> list[dict[str, Any]]:
    return read_jsonl(DATA_DERIVED / "manifest.jsonl")


@pytest.fixture(scope="module")
def meta() -> dict[str, Any]:
    out: dict[str, Any] = read_json(DATA_DERIVED / "manifest_meta.json")
    return out


@pytest.fixture(scope="module")
def audit() -> dict[str, Any]:
    out: dict[str, Any] = read_json(DATA_DERIVED / "audit.json")
    return out


def test_raw_checksums_match_pins() -> None:
    hashes = verify_raw(CFG, PROJECT_ROOT)
    assert hashes[CFG["source"]["dataset"]["local_file"]] == CFG["source"]["dataset"]["sha256"]


def test_rebuild_is_byte_identical_to_committed() -> None:
    check(CFG)


def test_sample_size_and_uniqueness(manifest: list[dict[str, Any]]) -> None:
    assert len(manifest) == 200
    assert len({m["pair_id"] for m in manifest}) == 200
    assert [m["manifest_index"] for m in manifest] == list(range(200))


def test_stratification(manifest: list[dict[str, Any]]) -> None:
    counts = Counter(m["category"] for m in manifest)
    assert counts == dict.fromkeys(CFG["sampling"]["categories"], 25)


def test_no_identical_candidates_and_turn_one(manifest: list[dict[str, Any]]) -> None:
    for m in manifest:
        c1, c2 = m["candidates"]["cand_1"], m["candidates"]["cand_2"]
        assert c1["text"] != c2["text"]
        assert c1["sha256"] < c2["sha256"]
        assert m["turn"] == 1


def test_hashes_recorded_in_meta(meta: dict[str, Any]) -> None:
    for name, digest in meta["artifacts_sha256"].items():
        assert sha256_file(DATA_DERIVED / name) == digest
    assert meta["source"]["raw_sha256"] == CFG["source"]["dataset"]["sha256"]
    assert meta["sampling_seed"] == CFG["sampling"]["seed"]
    assert meta["n_pairs"] == 200


def test_totals_reconcile(audit: dict[str, Any]) -> None:
    t = audit["totals"]
    assert t["source_rows"] == 3355
    pairs = read_jsonl(DATA_DERIVED / "pairs.jsonl")
    rows_in_pairs = sum(len(p["source_rows"]) for p in pairs)
    assert rows_in_pairs + sum(t["row_exclusions_by_reason"].values()) == t["source_rows"]
    assert len(audit["row_exclusions"]) == sum(t["row_exclusions_by_reason"].values())
    assert t["eligible_pairs"] == len(pairs)
    assert t["excluded_pairs_identical_text"] == len(audit["pair_exclusions"])
    assert sum(t["eligible_pool_human_aggregate"].values()) == len(pairs)


def test_aggregates_recompute_from_votes(manifest: list[dict[str, Any]]) -> None:
    for m in manifest:
        votes = [
            Vote(v["annotator"], v["vote"], v["raw_winner"], v["source_row"])
            for v in m["human"]["votes"]
        ]
        retained, _, aggregate, counts = aggregate_votes(m["pair_id"], votes)
        assert aggregate == m["human"]["aggregate"]
        assert counts == m["human"]["counts"]
        assert len(retained) == m["human"]["n_votes_retained"]


def test_every_vote_adjustment_is_reflected(
    audit: dict[str, Any], manifest: list[dict[str, Any]]
) -> None:
    by_pair = {p["pair_id"]: p for p in read_jsonl(DATA_DERIVED / "pairs.jsonl")}
    for adj in audit["vote_adjustments"]:
        votes = {v["source_row"]: v for v in by_pair[adj["pair_id"]]["human"]["votes"]}
        assert votes[adj["source_row"]]["retained"] is (adj["action"] == "kept")


def test_length_frozen(manifest: list[dict[str, Any]]) -> None:
    for m in manifest:
        for c in m["candidates"].values():
            assert c["whitespace_tokens"] == len(c["text"].split())
