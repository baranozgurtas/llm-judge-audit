"""Loading and hashing of the frozen study configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_judge_audit.io_utils import PROJECT_ROOT as PROJECT_ROOT
from llm_judge_audit.io_utils import read_json, sha256_file

CONFIG_PATH = PROJECT_ROOT / "config" / "study.json"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_DERIVED = PROJECT_ROOT / "data" / "derived"
RESULTS = PROJECT_ROOT / "results"
FREEZE_PATH = RESULTS / "freeze" / "study_freeze.json"
RUNS_DIR = RESULTS / "runs"
PREFLIGHT_DIR = RESULTS / "preflight"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

MANIFEST_FILE = "manifest.jsonl"
MANIFEST_META_FILE = "manifest_meta.json"
PAIRS_FILE = "pairs.jsonl"
AUDIT_FILE = "audit.json"
LICENSE_FILE = "dataset_license.json"
ATTRIBUTION_FILE = "ATTRIBUTION.md"

ORDERS: tuple[str, str] = ("original", "swapped")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    cfg = read_json(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cfg


def config_hash(path: Path = CONFIG_PATH) -> str:
    """SHA-256 of the exact config bytes on disk."""
    return sha256_file(path)
