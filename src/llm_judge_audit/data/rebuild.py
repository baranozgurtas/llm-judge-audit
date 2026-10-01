"""Deterministic rebuild of derived data artifacts from the pinned raw sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_judge_audit.config import DATA_DERIVED, PROJECT_ROOT
from llm_judge_audit.data.build import build_artifacts
from llm_judge_audit.data.download import verify_raw
from llm_judge_audit.io_utils import atomic_write_bytes, sha256_bytes


class RebuildMismatch(RuntimeError):
    pass


def rebuild(
    cfg: dict[str, Any], out_dir: Path = DATA_DERIVED, root: Path = PROJECT_ROOT
) -> dict[str, str]:
    """Build twice in memory, require byte identity, then write. Returns {file: sha256}."""
    verify_raw(cfg, root)
    first, _ = build_artifacts(cfg, root)
    second, _ = build_artifacts(cfg, root)
    if first != second:
        diff = sorted(k for k in first if first[k] != second.get(k))
        raise RebuildMismatch(f"non-deterministic build: {diff}")
    for name, data in sorted(first.items()):
        atomic_write_bytes(out_dir / name, data)
    return {name: sha256_bytes(data) for name, data in sorted(first.items())}


def check(cfg: dict[str, Any], out_dir: Path = DATA_DERIVED, root: Path = PROJECT_ROOT) -> None:
    """Rebuild in memory and require byte identity with the files on disk."""
    verify_raw(cfg, root)
    built, _ = build_artifacts(cfg, root)
    problems = []
    for name, data in sorted(built.items()):
        path = out_dir / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif path.read_bytes() != data:
            problems.append(f"{name}: differs from rebuild")
    if problems:
        raise RebuildMismatch("; ".join(problems))
