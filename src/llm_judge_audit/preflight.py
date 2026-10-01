"""S3 synthetic probes (at most two calls) and the S4 preflight report.

Probe inputs are synthetic SAMPLE DATA text, never benchmark pairs; outputs are stored under
results/preflight/, separate from benchmark runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from llm_judge_audit.config import DATA_DERIVED, LICENSE_FILE, MANIFEST_FILE, PREFLIGHT_DIR
from llm_judge_audit.data.download import ChecksumError, verify_raw
from llm_judge_audit.integrity import STATUSES, record_hash
from llm_judge_audit.io_utils import read_json, read_jsonl, sha256_file, write_json
from llm_judge_audit.ollama_client import JudgeClient
from llm_judge_audit.prompt import render_cell
from llm_judge_audit.runner import append_record, build_record, utc_now

PROBES_FILE = "probes.jsonl"
PROBE_SUMMARY_FILE = "probe_summary.json"
REPORT_FILE = "preflight_report.json"
MAX_PROBES = 2


def _synthetic_text(n_words: int, topic: str) -> str:
    words: list[str] = []
    i = 0
    while len(words) < n_words:
        words.extend(f"SAMPLE DATA sentence {i} about {topic} with neutral filler content.".split())
        i += 1
    return " ".join(words[:n_words])


def synthetic_probe_pairs(long_words: int) -> list[dict[str, Any]]:
    """Two synthetic pairs: a short one and one near the sample's longest combined length."""
    short = {
        "pair_id": "probe_short",
        "question": "SAMPLE DATA: What is 2 + 2? Answer briefly.",
        "candidates": {
            "cand_1": {"text": "2 + 2 = 4."},
            "cand_2": {"text": "2 + 2 = 5, because adding two and two gives five."},
        },
    }
    half = max(1, long_words // 2)
    long = {
        "pair_id": "probe_long",
        "question": "SAMPLE DATA: Write a detailed explanation of a synthetic topic.",
        "candidates": {
            "cand_1": {"text": _synthetic_text(half, "topic one")},
            "cand_2": {"text": _synthetic_text(half, "topic two")},
        },
    }
    return [short, long]


def run_probes(
    client: JudgeClient,
    freeze: dict[str, Any],
    num_ctx: int,
    manifest: list[dict[str, Any]],
    out_dir: Path = PREFLIGHT_DIR,
) -> list[dict[str, Any]]:
    out = out_dir / PROBES_FILE
    if out.exists() and read_jsonl(out):
        raise RuntimeError(
            f"probes already recorded in {out}; the budget is {MAX_PROBES} calls in total"
        )
    longest = max(
        len(m["question"].split()) + sum(c["whitespace_tokens"] for c in m["candidates"].values())
        for m in manifest
    )
    identity = {
        "run_id": "preflight-synthetic",
        "study_hash": freeze["study_hash"],
        "config_sha256": freeze["config_sha256"],
        "prompt_sha256": freeze["prompt_sha256"],
        "parser_version": freeze["parser_version"],
        "parser_sha256": freeze["parser_sha256"],
        "manifest_sha256": "SAMPLE DATA (synthetic probe; not a manifest pair)",
        "model_tag": freeze["model_tag"],
        "model_digest": freeze["model_digest"],
        "ollama_version": freeze["ollama_version"],
    }
    records = []
    for pair in synthetic_probe_pairs(longest):
        rendered = render_cell(pair, "original")
        result = client.chat(rendered.prompt)
        rec = build_record(
            identity,
            pair["pair_id"],
            "original",
            rendered.mapping,
            rendered.prompt,
            result,
            num_ctx,
        )
        rec["label"] = "SAMPLE DATA - synthetic preflight probe, not benchmark data"
        rec["rendered_prompt_chars"] = len(rendered.prompt)
        rec["record_sha256"] = record_hash(rec)
        append_record(out, rec)
        records.append(rec)
    return records


def estimate_runtime(
    probes: list[dict[str, Any]], manifest: list[dict[str, Any]]
) -> dict[str, Any]:
    """Uncertain 400-cell runtime range from probe throughput and manifest prompt sizes."""
    ok = [p for p in probes if p["prompt_tokens"] and p["runtime"].get("prompt_eval_duration")]
    if not ok:
        return {"available": False, "reason": "no probe reported token timings"}
    prefill = [p["runtime"]["prompt_eval_duration"] / 1e9 / p["prompt_tokens"] for p in ok]
    decode = [p["runtime"]["eval_duration"] / 1e9 / max(1, p["completion_tokens"]) for p in ok]
    cpt = min(p["rendered_prompt_chars"] / p["prompt_tokens"] for p in ok)
    est_tokens = sorted(
        len(render_cell(m, o).prompt) / cpt for m in manifest for o in ("original", "swapped")
    )
    total_prompt = sum(est_tokens)
    completions = [p["completion_tokens"] for p in ok]
    n = len(est_tokens)
    low = total_prompt * min(prefill) + n * min(completions) * min(decode)
    high = (total_prompt * max(prefill) + n * max(*completions, 120) * max(decode)) * 1.5
    return {
        "available": True,
        "chars_per_prompt_token": cpt,
        "estimated_prompt_tokens": {
            "per_cell_max": round(est_tokens[-1]),
            "per_cell_median": round(est_tokens[len(est_tokens) // 2]),
            "total": round(total_prompt),
            "cells": len(est_tokens),
        },
        "prefill_s_per_token": prefill,
        "decode_s_per_token": decode,
        "runtime_hours_range": [round(low / 3600, 2), round(high / 3600, 2)],
        "method": "sum over 400 cells of est. prompt tokens x prefill rate + completion tokens x "
        "decode rate; low = fastest probe rates & shortest completion, high = slowest rates, "
        ">=120 completion tokens, x1.5 margin. Excludes model load and thermal throttling.",
    }


def run_tests() -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if " passed" in ln or " failed" in ln]
    summary = lines[-1].strip() if lines else (proc.stdout + proc.stderr)[-300:]
    return {"passed": proc.returncode == 0, "summary": summary}


def build_report(
    cfg: dict[str, Any],
    freeze: dict[str, Any],
    live: dict[str, Any],
    tests: dict[str, Any],
    freeze_ok: bool,
    freeze_error: str | None,
    out_dir: Path = PREFLIGHT_DIR,
) -> dict[str, Any]:
    manifest_path = DATA_DERIVED / MANIFEST_FILE
    manifest = read_jsonl(manifest_path)
    license_doc = read_json(DATA_DERIVED / LICENSE_FILE)
    probes = read_jsonl(out_dir / PROBES_FILE) if (out_dir / PROBES_FILE).exists() else []
    try:
        verify_raw(cfg)
        raw_ok = True
    except (OSError, ChecksumError):
        raw_ok = False
    probe_ok = bool(probes) and all(p["status"] == "valid" for p in probes)
    checks = {
        "dataset_license_verified": license_doc["dataset"]["license"] == "CC-BY-4.0",
        "raw_checksums_match": raw_ok,
        "sample_n_200": len(manifest) == 200,
        "manifest_hash_matches_freeze": sha256_file(manifest_path) == freeze["manifest_sha256"],
        "freeze_matches_code": freeze_ok,
        "live_model_digest_matches": live.get("model_digest") == freeze["model_digest"],
        "live_ollama_version_matches": live.get("ollama_version") == freeze["ollama_version"],
        "synthetic_probes_valid": probe_ok,
        "offline_tests_pass": tests["passed"],
    }
    report = {
        "generated_utc": utc_now(),
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "freeze_error": freeze_error,
        "dataset": {
            "repo": cfg["source"]["dataset"]["hf_repo"],
            "revision": cfg["source"]["dataset"]["hf_revision"],
            "license": license_doc["dataset"]["license"],
        },
        "sample": {
            "n_pairs": len(manifest),
            "expected_cells": 2 * len(manifest),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "model": {
            "tag": freeze["model_tag"],
            "digest": freeze["model_digest"],
            "ollama_version": freeze["ollama_version"],
            "live": live,
        },
        "decoding": freeze["decoding"],
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
        "probes": [
            {
                "pair_id": p["pair_id"],
                "status": p["status"],
                "latency_s": p["latency_s"],
                "prompt_tokens": p["prompt_tokens"],
                "completion_tokens": p["completion_tokens"],
                "raw_response": p["raw_response"],
            }
            for p in probes
        ],
        "runtime_estimate": estimate_runtime(probes, manifest),
        "tests": tests,
        "status_vocabulary": list(STATUSES),
    }
    write_json(out_dir / REPORT_FILE, report)
    return report
