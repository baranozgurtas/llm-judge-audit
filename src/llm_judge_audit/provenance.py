"""Machine, git and runtime provenance (sanitised: no environment variables or secrets)."""

from __future__ import annotations

import platform
import subprocess
from typing import Any

from llm_judge_audit.config import PROJECT_ROOT


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    commit = _run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "dirty_files": None if not status else len(status.splitlines()),
    }


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        info["model"] = _run(["sysctl", "-n", "hw.model"])
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["cpu_cores"] = _run(["sysctl", "-n", "hw.ncpu"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        info["memory_gb"] = round(int(mem) / 2**30, 1) if mem and mem.isdigit() else None
    return info
