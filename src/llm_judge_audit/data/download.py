"""Download pinned raw sources and verify their SHA-256 checksums.

Only the `human` split parquet is fetched; the GPT-4-judgment split is never downloaded.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

from llm_judge_audit.config import PROJECT_ROOT
from llm_judge_audit.io_utils import atomic_write_bytes, sha256_bytes, sha256_file


class ChecksumError(RuntimeError):
    pass


def _sources(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    src = cfg["source"]
    return [src["dataset"], src["questions"]]


def verify_raw(cfg: dict[str, Any], root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Return {local_file: sha256}; raise if any file is missing or mismatched."""
    out: dict[str, str] = {}
    for spec in _sources(cfg):
        path = root / spec["local_file"]
        if not path.exists():
            raise FileNotFoundError(f"raw file missing: {spec['local_file']} (run `lja download`)")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise ChecksumError(
                f"checksum mismatch for {spec['local_file']}: "
                f"expected {spec['sha256']}, got {digest}"
            )
        out[spec["local_file"]] = digest
    return out


def download(cfg: dict[str, Any], root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, str]:
    """Fetch each pinned source if absent, verifying the checksum before writing."""
    for spec in _sources(cfg):
        path = root / spec["local_file"]
        if path.exists() and not force:
            continue
        if "gpt4" in spec["url"].lower():
            raise RuntimeError("refusing to download the GPT-4-judgment split")
        with urllib.request.urlopen(spec["url"], timeout=120) as resp:
            data = resp.read()
        digest = sha256_bytes(data)
        if digest != spec["sha256"]:
            raise ChecksumError(
                f"downloaded {spec['url']} has sha256 {digest}, expected {spec['sha256']}"
            )
        atomic_write_bytes(path, data)
    return verify_raw(cfg, root)
