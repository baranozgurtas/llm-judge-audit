"""Post-run analysis: integrity report, registered metrics, provenance, error-review selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_judge_audit.error_review import select_error_cases
from llm_judge_audit.integrity import check_integrity, public_report
from llm_judge_audit.io_utils import read_json, read_jsonl, sha256_file, write_json
from llm_judge_audit.metrics import compute_metrics
from llm_judge_audit.provenance import git_state
from llm_judge_audit.runner import CELLS_FILE, RUN_META_FILE, SESSIONS_FILE

INTEGRITY_FILE = "integrity.json"
METRICS_FILE = "metrics.json"
PROVENANCE_FILE = "provenance.json"
ERROR_REVIEW_FILE = "error_review.json"


class AnalysisRefused(RuntimeError):
    pass


def _load(run_dir: Path, manifest: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = read_json(run_dir / RUN_META_FILE)
    report = check_integrity(
        run_dir / CELLS_FILE, [m["pair_id"] for m in manifest], meta["identity"]
    )
    return meta, report


def analyze_run(
    run_dir: Path, manifest: list[dict[str, Any]], cfg: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    meta, report = _load(run_dir, manifest)
    if sha256_file(manifest_path) != meta["identity"]["manifest_sha256"]:
        raise AnalysisRefused("manifest on disk differs from the run's manifest hash")
    write_json(run_dir / INTEGRITY_FILE, public_report(report))
    if not report["integrity_ok"]:
        raise AnalysisRefused(
            f"integrity failed ({report['completion_state']}); metrics not computed"
        )
    sessions = read_jsonl(run_dir / SESSIONS_FILE) if (run_dir / SESSIONS_FILE).exists() else []
    metrics = compute_metrics(
        manifest, report["accepted"], report["counts"], sessions, cfg["analysis"]
    )
    state = report["completion_state"]
    metrics["completion_state"] = state
    metrics["run_id"] = meta["identity"]["run_id"]
    metrics["scope_note"] = (
        "Benchmark complete: all 400 cells valid and integrity checks passed."
        if state == "COMPLETE"
        else f"{state}: descriptive metrics over the cells available; not a completed benchmark."
    )
    write_json(run_dir / METRICS_FILE, metrics)
    write_json(run_dir / PROVENANCE_FILE, build_provenance(meta, report, sessions, cfg))
    return metrics


def build_provenance(
    meta: dict[str, Any],
    report: dict[str, Any],
    sessions: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    freeze = meta["freeze"]
    ds, qs = cfg["source"]["dataset"], cfg["source"]["questions"]
    starts = [s["session_start_utc"] for s in sessions if s.get("session_start_utc")]
    ends = [s["session_end_utc"] for s in sessions if s.get("session_end_utc")]
    return {
        "run_id": meta["identity"]["run_id"],
        "completion_state": report["completion_state"],
        "utc_start": min(starts) if starts else None,
        "utc_end": max(ends) if ends else None,
        "run_created_utc": meta.get("created_utc"),
        "sessions": sessions,
        "git_at_run_start": meta.get("git"),
        "git_at_analysis": git_state(),
        "source": {
            "dataset": ds["hf_repo"],
            "revision": ds["hf_revision"],
            "split": ds["split"],
            "license": ds["license"],
            "raw_sha256": ds["sha256"],
            "questions_repo": qs["repo"],
            "questions_revision": qs["revision"],
            "questions_sha256": qs["sha256"],
        },
        "manifest_sha256": freeze["manifest_sha256"],
        "sampling": {
            "seed": freeze["sampling_seed"],
            "rule": freeze["sampling_rule_version"],
            "aggregation_rule": freeze["aggregation_rule_version"],
        },
        "model": {
            "tag": freeze["model_tag"],
            "digest": freeze["model_digest"],
            "ollama_version": freeze["ollama_version"],
            "details": meta.get("model_details"),
        },
        "hashes": {
            k: freeze[k]
            for k in (
                "study_hash",
                "prompt_sha256",
                "parser_sha256",
                "metrics_sha256",
                "config_sha256",
            )
        },
        "versions": {
            k: freeze[k]
            for k in ("prompt_version", "parser_version", "metrics_version", "config_version")
        },
        "decoding": freeze["decoding"],
        "hardware": meta.get("hardware"),
        "cell_status_counts": report["counts"],
    }


def run_error_review(
    run_dir: Path, manifest: list[dict[str, Any]], cfg: dict[str, Any]
) -> dict[str, Any]:
    if not (run_dir / METRICS_FILE).exists():
        raise AnalysisRefused("primary metrics must be saved before error review")
    if (run_dir / ERROR_REVIEW_FILE).exists():
        raise AnalysisRefused("error review already selected; selection is never redrawn")
    _, report = _load(run_dir, manifest)
    if not report["integrity_ok"]:
        raise AnalysisRefused("integrity failed; error review not selected")
    review = select_error_cases(
        manifest,
        report["accepted"],
        int(cfg["analysis"]["error_review_seed"]),
        int(cfg["analysis"]["error_review_max_cases"]),
    )
    review["completion_state"] = report["completion_state"]
    write_json(run_dir / ERROR_REVIEW_FILE, review)
    return review


SUPPLEMENTARY_DIR = "supplementary"
SUPPLEMENTARY_FILE = "supplementary_analysis_v1.json"


def run_supplementary(
    run_dir: Path, manifest: list[dict[str, Any]], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Post-hoc supplementary analysis from immutable saved records; never edits metrics.json."""
    from llm_judge_audit.supplementary import supplementary_analysis

    if not (run_dir / METRICS_FILE).exists():
        raise AnalysisRefused("registered metrics must exist before supplementary analysis")
    _, report = _load(run_dir, manifest)
    if not report["integrity_ok"]:
        raise AnalysisRefused("integrity failed; supplementary analysis not computed")
    registered = read_json(run_dir / METRICS_FILE)
    sup = supplementary_analysis(manifest, report["accepted"], registered, cfg["analysis"])
    sup["inputs_sha256"] = {
        CELLS_FILE: sha256_file(run_dir / CELLS_FILE),
        METRICS_FILE: sha256_file(run_dir / METRICS_FILE),
    }
    write_json(run_dir / SUPPLEMENTARY_DIR / SUPPLEMENTARY_FILE, sup)
    return sup
