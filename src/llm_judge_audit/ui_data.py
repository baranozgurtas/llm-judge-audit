"""Read-only loading of saved artifacts for the dashboard. Never imports the runner or client."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_judge_audit.config import (
    DATA_DERIVED,
    FREEZE_PATH,
    MANIFEST_FILE,
    MANIFEST_META_FILE,
    PREFLIGHT_DIR,
    RUNS_DIR,
    SAMPLE_DATA_DIR,
)
from llm_judge_audit.io_utils import read_json, read_jsonl

SAMPLE_SOURCE = "SAMPLE DATA (synthetic demo)"


@dataclass
class Bundle:
    source: str
    is_sample: bool
    manifest: list[dict[str, Any]]
    manifest_meta: dict[str, Any] | None
    run_dir: Path | None
    run_meta: dict[str, Any] | None = None
    integrity: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    error_review: dict[str, Any] | None = None
    cells: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    freeze: dict[str, Any] | None = None
    preflight: dict[str, Any] | None = None
    supplementary: dict[str, Any] | None = None


def _opt_json(path: Path) -> Any:
    return read_json(path) if path.exists() else None


def list_sources() -> list[str]:
    runs = sorted(p.name for p in RUNS_DIR.glob("*") if (p / "run_meta.json").exists())
    return [*runs, SAMPLE_SOURCE] if runs else ["(no benchmark run yet)", SAMPLE_SOURCE]


def load_cells(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """First record per (pair, order); tolerant read for display (integrity is reported apart)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return out
    import json

    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and "pair_id" in rec and "order" in rec:
            out.setdefault((rec["pair_id"], rec["order"]), rec)
    return out


def load_bundle(source: str) -> Bundle:
    if source == SAMPLE_SOURCE:
        run_dir: Path | None = SAMPLE_DATA_DIR / "run-SAMPLE-DATA"
        manifest_path = SAMPLE_DATA_DIR / "manifest.jsonl"
        meta = None
        is_sample = True
    else:
        run_dir = RUNS_DIR / source if (RUNS_DIR / source).exists() else None
        manifest_path = DATA_DERIVED / MANIFEST_FILE
        meta = _opt_json(DATA_DERIVED / MANIFEST_META_FILE)
        is_sample = False
    manifest = read_jsonl(manifest_path) if manifest_path.exists() else []
    b = Bundle(source, is_sample, manifest, meta, run_dir)
    if not is_sample:
        b.freeze = _opt_json(FREEZE_PATH)
        b.preflight = _opt_json(PREFLIGHT_DIR / "preflight_report.json")
    if run_dir is not None and run_dir.exists():
        b.run_meta = _opt_json(run_dir / "run_meta.json")
        b.integrity = _opt_json(run_dir / "integrity.json")
        b.metrics = _opt_json(run_dir / "metrics.json")
        b.provenance = _opt_json(run_dir / "provenance.json")
        b.error_review = _opt_json(run_dir / "error_review.json")
        b.cells = load_cells(run_dir / "cells.jsonl")
        b.supplementary = _opt_json(run_dir / "supplementary" / "supplementary_analysis_v1.json")
    return b


def fmt_rate(r: dict[str, Any] | None, digits: int = 1) -> str:
    """'k/n = x% (95% CI lo–hi%)' with explicit numerator and denominator."""
    if not r:
        return "not available"
    k, n = r["numerator"], r["denominator"]
    if not n:
        return f"{k}/{n} (undefined)"
    s = f"{k}/{n} = {r['rate']:.{digits}%}"
    if r.get("wilson95"):
        lo, hi = r["wilson95"]
        s += f" (Wilson 95% CI {lo:.{digits}%}–{hi:.{digits}%})"
    return s


def fmt_boot(b: dict[str, Any] | None, pct: bool = False) -> str:
    if not b or b.get("estimate") is None:
        return "not available"
    est = b["estimate"]

    def f(x: float) -> str:
        return f"{x:.1%}" if pct else f"{x:.3f}"

    s = f(est)
    if b.get("ci95"):
        lo, hi = b["ci95"]
        s += f" (pair-bootstrap 95% CI {f(lo)} to {f(hi)}; {b['resamples']} resamples)"
    return s
