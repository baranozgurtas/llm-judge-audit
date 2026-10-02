"""Command-line entry point: `uv run lja <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_judge_audit.config import (
    DATA_DERIVED,
    FREEZE_PATH,
    MANIFEST_FILE,
    PREFLIGHT_DIR,
    RUNS_DIR,
    load_config,
)
from llm_judge_audit.io_utils import read_json, read_jsonl


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _manifest() -> list[dict[str, Any]]:
    return read_jsonl(DATA_DERIVED / MANIFEST_FILE)


def _live(cfg: dict[str, Any]) -> dict[str, Any]:
    from llm_judge_audit.ollama_client import model_digest, ollama_version

    endpoint, tag = cfg["judge"]["endpoint"], cfg["judge"]["model_tag"]
    try:
        return {
            "ollama_version": ollama_version(endpoint),
            "model_digest": model_digest(endpoint, tag),
        }
    except OSError as exc:
        return {"error": f"ollama unreachable: {exc!r}"}


def _run_dir(args: argparse.Namespace) -> Path:
    if args.run_id:
        return RUNS_DIR / str(args.run_id)
    from llm_judge_audit.freeze import run_id_for

    study_hash: str = read_json(FREEZE_PATH)["study_hash"]
    return RUNS_DIR / run_id_for(study_hash)


def cmd_download(args: argparse.Namespace) -> int:
    from llm_judge_audit.data.download import download

    _print(download(load_config(), force=args.force))
    return 0


def cmd_rebuild_data(args: argparse.Namespace) -> int:
    from llm_judge_audit.data import rebuild

    cfg = load_config()
    if args.check:
        rebuild.check(cfg)
        print("OK: derived artifacts are byte-identical to a fresh rebuild")
        return 0
    _print(rebuild.rebuild(cfg))
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    from llm_judge_audit.freeze import write_freeze

    _print(write_freeze())
    return 0


def cmd_verify_freeze(args: argparse.Namespace) -> int:
    from llm_judge_audit.freeze import verify_freeze

    frozen = verify_freeze()
    live = _live(load_config())
    ok = all(live.get(k) == frozen[k] for k in ("model_digest", "ollama_version"))
    _print(
        {
            "freeze": "MATCH",
            "study_hash": frozen["study_hash"],
            "live": live,
            "live_matches_freeze": ok,
        }
    )
    return 0 if ok else 1


def cmd_probe(args: argparse.Namespace) -> int:
    from llm_judge_audit.freeze import verify_freeze
    from llm_judge_audit.ollama_client import OllamaJudge, model_details
    from llm_judge_audit.preflight import run_probes
    from llm_judge_audit.provenance import hardware

    cfg = load_config()
    frozen = verify_freeze()
    live = _live(cfg)
    if live.get("model_digest") != frozen["model_digest"]:
        print(f"REFUSED: live model {live} does not match frozen digest", file=sys.stderr)
        return 1
    records = run_probes(
        OllamaJudge(cfg["judge"]), frozen, cfg["judge"]["options"]["num_ctx"], _manifest()
    )
    from llm_judge_audit.io_utils import write_json

    write_json(
        PREFLIGHT_DIR / "probe_environment.json",
        {
            "label": "SAMPLE DATA - synthetic preflight probes",
            "live": live,
            "hardware": hardware(),
            "model_details": model_details(cfg["judge"]["endpoint"], cfg["judge"]["model_tag"]),
            "options": cfg["judge"]["options"],
        },
    )
    _print(
        [
            {
                k: r[k]
                for k in (
                    "pair_id",
                    "status",
                    "raw_response",
                    "latency_s",
                    "prompt_tokens",
                    "completion_tokens",
                    "done_reason",
                    "error",
                )
            }
            for r in records
        ]
    )
    return 0 if all(r["status"] == "valid" for r in records) else 1


def cmd_preflight(args: argparse.Namespace) -> int:
    from llm_judge_audit.freeze import FreezeError, verify_freeze
    from llm_judge_audit.preflight import build_report, run_tests

    cfg = load_config()
    frozen = read_json(FREEZE_PATH)
    try:
        verify_freeze()
        freeze_ok, freeze_err = True, None
    except FreezeError as exc:
        freeze_ok, freeze_err = False, str(exc)
    tests = (
        {"passed": True, "summary": "skipped (--skip-tests)"} if args.skip_tests else run_tests()
    )
    report = build_report(cfg, frozen, _live(cfg), tests, freeze_ok, freeze_err)
    _print(report)
    return 0 if report["verdict"] == "PASS" else 1


def cmd_run(args: argparse.Namespace) -> int:
    from llm_judge_audit.freeze import run_id_for, verify_freeze
    from llm_judge_audit.ollama_client import OllamaJudge, model_details
    from llm_judge_audit.provenance import git_state, hardware
    from llm_judge_audit.runner import run_benchmark

    if not args.approved:
        print(
            "REFUSED: benchmark inference requires explicit approval (--approved).", file=sys.stderr
        )
        return 2
    cfg = load_config()
    frozen = verify_freeze()
    report_path = PREFLIGHT_DIR / "preflight_report.json"
    if not report_path.exists():
        print("REFUSED: run `lja preflight` first", file=sys.stderr)
        return 2
    pre = read_json(report_path)
    if pre["verdict"] != "PASS" or pre["hashes"]["study_hash"] != frozen["study_hash"]:
        print("REFUSED: preflight is not PASS for the current freeze", file=sys.stderr)
        return 2
    judge = cfg["judge"]
    run_dir = RUNS_DIR / run_id_for(frozen["study_hash"])
    extra = {
        "git": git_state(),
        "hardware": hardware(),
        "model_details": model_details(judge["endpoint"], judge["model_tag"]),
    }

    def progress(i: int, total: int, rec: dict[str, Any]) -> None:
        print(
            f"[{i}/{total}] {rec['pair_id']} {rec['order']:<8} {rec['status']:<14} "
            f"{rec['mapped_verdict']!s:<7} {rec['latency_s']:.1f}s",
            flush=True,
        )

    report = run_benchmark(
        run_dir,
        _manifest(),
        frozen,
        _live(cfg),
        OllamaJudge(judge),
        judge["options"]["num_ctx"],
        judge["max_consecutive_connection_errors"],
        extra,
        progress,
    )
    _print(
        {
            "run_dir": str(run_dir),
            "completion_state": report["completion_state"],
            "counts": report["counts"],
            "aborted": report["aborted"],
        }
    )
    return 0


def cmd_integrity(args: argparse.Namespace) -> int:
    from llm_judge_audit.integrity import check_integrity, public_report

    run_dir = _run_dir(args)
    meta = read_json(run_dir / "run_meta.json")
    report = public_report(
        check_integrity(
            run_dir / "cells.jsonl", [m["pair_id"] for m in _manifest()], meta["identity"]
        )
    )
    report.pop("missing_cells")
    _print(report)
    return 0 if report["integrity_ok"] else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    from llm_judge_audit.analysis import analyze_run
    from llm_judge_audit.freeze import verify_freeze

    verify_freeze()
    metrics = analyze_run(_run_dir(args), _manifest(), load_config(), DATA_DERIVED / MANIFEST_FILE)
    _print(
        {
            "completion_state": metrics["completion_state"],
            "main_accuracy": metrics["human_agreement"]["main_accuracy"]["rate"],
            "cell_counts": metrics["operational"]["cell_counts"],
        }
    )
    return 0


def cmd_error_review(args: argparse.Namespace) -> int:
    from llm_judge_audit.analysis import run_error_review

    review = run_error_review(_run_dir(args), _manifest(), load_config())
    _print({k: review[k] for k in ("eligible_incorrect_pairs", "selected_pair_ids")})
    return 0


def cmd_supplementary(args: argparse.Namespace) -> int:
    from llm_judge_audit.analysis import run_supplementary

    sup = run_supplementary(_run_dir(args), _manifest(), load_config())
    _print(
        {
            "verbosity_checks": sup["verbosity"]["checks"],
            "bootstrap_reproduction": {
                k: v["matches"] for k, v in sup["registered_bootstrap_reproduction"].items()
            },
        }
    )
    return 0


def cmd_make_sample_data(args: argparse.Namespace) -> int:
    from llm_judge_audit.sample_data import make_sample_run

    _print({"written": str(make_sample_run())})
    return 0


COMMANDS: dict[str, tuple[str, Callable[[argparse.Namespace], int]]] = {
    "download": ("download pinned raw sources and verify checksums", cmd_download),
    "rebuild-data": (
        "deterministically rebuild derived data from pinned raw data",
        cmd_rebuild_data,
    ),
    "freeze": ("freeze prompt/parser/metrics/config/manifest/model hashes", cmd_freeze),
    "verify-freeze": ("verify code and live Ollama match the freeze", cmd_verify_freeze),
    "probe": ("S3: at most two synthetic Gemma probe calls (not benchmark data)", cmd_probe),
    "preflight": ("S4: compact PASS/FAIL preflight report (no inference)", cmd_preflight),
    "run": ("S4: run/resume the 400-cell benchmark (requires --approved)", cmd_run),
    "integrity": ("validate a run's cell log", cmd_integrity),
    "analyze": ("integrity + registered metrics + provenance for a run", cmd_analyze),
    "error-review": ("seeded selection of incorrect resolved pairs", cmd_error_review),
    "supplementary": (
        "post-hoc supplementary-analysis-v1 (verbosity + interval audit); no inference",
        cmd_supplementary,
    ),
    "make-sample-data": ("write the synthetic SAMPLE DATA run for UI demos", cmd_make_sample_data),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lja", description="LLM Judge Audit")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, func) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        if name == "download":
            p.add_argument("--force", action="store_true", help="re-download even if present")
        if name == "rebuild-data":
            p.add_argument(
                "--check",
                action="store_true",
                help="verify on-disk artifacts equal a fresh rebuild; write nothing",
            )
        if name == "preflight":
            p.add_argument("--skip-tests", action="store_true", help="do not run pytest")
        if name == "run":
            p.add_argument(
                "--approved",
                action="store_true",
                help="explicit user approval for the 400 benchmark calls",
            )
        if name in ("integrity", "analyze", "error-review", "supplementary"):
            p.add_argument("--run-id", default=None, help="defaults to the frozen study's run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
