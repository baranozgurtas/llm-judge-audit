"""Study freeze: hashes of every component that must not change between cells of one run."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_judge_audit.config import (
    CONFIG_PATH,
    DATA_DERIVED,
    FREEZE_PATH,
    MANIFEST_FILE,
    PROJECT_ROOT,
    RUNS_DIR,
    load_config,
)
from llm_judge_audit.io_utils import read_json, sha256_file, sha256_json, write_json
from llm_judge_audit.parser import OUTPUT_SCHEMA, PARSER_VERSION
from llm_judge_audit.prompt import PROMPT_PATH, PROMPT_VERSION, prompt_hash

FREEZE_VERSION = "freeze-v1"
SRC = PROJECT_ROOT / "src" / "llm_judge_audit"
PARSER_FILES = [SRC / "parser.py"]
METRICS_FILES = [SRC / "metrics.py", SRC / "stats.py"]


class FreezeError(RuntimeError):
    pass


def _files_hash(paths: list[Path]) -> str:
    return sha256_json({p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in paths})


def compute_study_hashes(
    cfg: dict[str, Any] | None = None, manifest_path: Path = DATA_DERIVED / MANIFEST_FILE
) -> dict[str, Any]:
    from llm_judge_audit.metrics import METRICS_VERSION

    cfg = load_config() if cfg is None else cfg
    judge = cfg["judge"]
    components: dict[str, Any] = {
        "freeze_version": FREEZE_VERSION,
        "config_version": cfg["config_version"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "prompt_version": PROMPT_VERSION,
        "prompt_file": PROMPT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "prompt_sha256": prompt_hash(),
        "parser_version": PARSER_VERSION,
        "parser_sha256": _files_hash(PARSER_FILES),
        "output_contract": OUTPUT_SCHEMA,
        "metrics_version": METRICS_VERSION,
        "metrics_sha256": _files_hash(METRICS_FILES),
        "manifest_sha256": sha256_file(manifest_path),
        "sampling_seed": cfg["sampling"]["seed"],
        "sampling_rule_version": cfg["sampling"]["rule_version"],
        "aggregation_rule_version": cfg["aggregation"]["rule_version"],
        "source_revision": cfg["source"]["dataset"]["hf_revision"],
        "raw_sha256": cfg["source"]["dataset"]["sha256"],
        "model_tag": judge["model_tag"],
        "model_digest": judge["model_digest"],
        "ollama_version": judge["ollama_version"],
        "decoding": {
            "options": judge["options"],
            "format": "json-schema (output_contract)",
            "keep_alive": judge["keep_alive"],
            "request_timeout_s": judge["request_timeout_s"],
            "retries": 0,
        },
    }
    components["study_hash"] = sha256_json(components)
    return components


def run_id_for(study_hash: str) -> str:
    return f"bench-{study_hash[:12]}"


def write_freeze(path: Path = FREEZE_PATH, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    hashes = compute_study_hashes()
    if path.exists():
        old = read_json(path)
        if old["study_hash"] == hashes["study_hash"]:
            return dict(old)
        if (runs_dir / run_id_for(old["study_hash"])).exists():
            raise FreezeError(
                "a run already exists for the current freeze; changes require new versions and "
                "a separate run (move the old freeze aside deliberately)"
            )
    doc = {**hashes, "frozen_at_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    write_json(path, doc)
    return doc


def verify_freeze(path: Path = FREEZE_PATH) -> dict[str, Any]:
    """Recompute hashes and require exact equality with the freeze file."""
    if not path.exists():
        raise FreezeError(f"no freeze file at {path}; run `lja freeze` first")
    frozen = read_json(path)
    current = compute_study_hashes()
    diffs = sorted(k for k in current if current[k] != frozen.get(k))
    if diffs:
        raise FreezeError(f"study components changed since freeze: {diffs}")
    return dict(frozen)
