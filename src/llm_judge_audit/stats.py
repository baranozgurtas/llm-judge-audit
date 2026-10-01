"""Statistical helpers: Wilson intervals, Cohen's kappa, pair-cluster bootstrap."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> dict[str, Any]:
    """Count, denominator, rate and Wilson 95% interval (None when n == 0)."""
    ci = wilson(k, n)
    return {
        "numerator": k,
        "denominator": n,
        "rate": (k / n) if n else None,
        "wilson95": list(ci) if ci else None,
    }


def cohen_kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str]) -> float | None:
    """Unweighted Cohen's kappa; None when undefined (no items or chance agreement = 1)."""
    if len(a) != len(b):
        raise ValueError("length mismatch")
    n = len(a)
    if n == 0:
        return None
    p_o = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum((ca[lab] / n) * (cb[lab] / n) for lab in labels)
    if math.isclose(p_e, 1.0):
        return None
    return (p_o - p_e) / (1 - p_e)


def cluster_bootstrap(
    cluster_ids: Sequence[str],
    statistic: Callable[[list[str]], float | None],
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Percentile 95% interval resampling whole clusters (pair IDs) with replacement.

    `statistic` receives a list of cluster ids (with repeats) and must evaluate every unit
    belonging to each listed cluster, so both orders of a pair always travel together.
    Resamples where the statistic is undefined are counted and skipped.
    """
    ids = sorted(cluster_ids)
    point = statistic(ids)
    if not ids or point is None:
        return {
            "estimate": point,
            "ci95": None,
            "resamples": resamples,
            "undefined": None,
            "seed": seed,
        }
    rng = np.random.default_rng(seed)
    values: list[float] = []
    undefined = 0
    for _ in range(resamples):
        idx = rng.integers(0, len(ids), size=len(ids))
        value = statistic([ids[i] for i in idx])
        if value is None:
            undefined += 1
        else:
            values.append(value)
    ci = None
    if values:
        lo, hi = np.percentile(np.array(values), [2.5, 97.5])
        ci = [float(lo), float(hi)]
    return {
        "estimate": point,
        "ci95": ci,
        "resamples": resamples,
        "undefined": undefined,
        "seed": seed,
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy default); None for empty input."""
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), q))
