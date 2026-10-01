"""Seeded selection of incorrect resolved pairs for qualitative error review.

Selection rule (error-review-v1, predeclared): candidates are resolved pairs where at least one
valid cell's mapped verdict differs from the human aggregate. Rank by
sha256('{seed}|{pair_id}') ascending; take the first `max_cases`. Selection only runs after the
integrity-checked cell log and primary metrics are saved. Observations are added afterwards
and never change labels, prompts, or metrics.
"""

from __future__ import annotations

from typing import Any

from llm_judge_audit.io_utils import sha256_text

ERROR_REVIEW_VERSION = "error-review-v1"


def select_error_cases(
    manifest: list[dict[str, Any]],
    accepted: dict[tuple[str, str], dict[str, Any]],
    seed: int,
    max_cases: int,
) -> dict[str, Any]:
    candidates = []
    for m in manifest:
        human = m["human"]["aggregate"]
        if human not in ("cand_1", "cand_2"):
            continue
        cells = [accepted.get((m["pair_id"], o)) for o in ("original", "swapped")]
        valid = [c for c in cells if c is not None and c["status"] == "valid"]
        if any(c["mapped_verdict"] != human for c in valid):
            candidates.append(m)
    ranked = sorted(candidates, key=lambda m: sha256_text(f"{seed}|{m['pair_id']}"))
    selected = ranked[:max_cases]
    cases = []
    for m in selected:
        cases.append(
            {
                "pair_id": m["pair_id"],
                "category": m["category"],
                "human_aggregate": m["human"]["aggregate"],
                "human_counts": m["human"]["counts"],
                "whitespace_tokens": {
                    k: c["whitespace_tokens"] for k, c in m["candidates"].items()
                },
                "judge": {
                    o: {
                        "status": rec["status"],
                        "displayed": rec["verdict_displayed"],
                        "mapped": rec["mapped_verdict"],
                        "rationale": (rec["parse"] or {}).get("rationale"),
                    }
                    for o in ("original", "swapped")
                    if (rec := accepted.get((m["pair_id"], o))) is not None
                },
                "observation": None,
            }
        )
    return {
        "error_review_version": ERROR_REVIEW_VERSION,
        "rule": __doc__,
        "seed": seed,
        "max_cases": max_cases,
        "eligible_incorrect_pairs": len(candidates),
        "selected_pair_ids": [m["pair_id"] for m in selected],
        "cases": cases,
    }
