"""SAMPLE DATA: synthetic manifest, fake judge and demo run. Never real data or model output."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from llm_judge_audit.analysis import analyze_run, run_error_review
from llm_judge_audit.config import SAMPLE_DATA_DIR, load_config
from llm_judge_audit.io_utils import sha256_file, sha256_text, write_jsonl
from llm_judge_audit.ollama_client import ChatResult
from llm_judge_audit.runner import run_benchmark

SAMPLE_LABEL = "SAMPLE DATA"
CATEGORIES = [
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
]


def synthetic_manifest(n_pairs: int = 24) -> list[dict[str, Any]]:
    labels = ["cand_1", "cand_2", "cand_1", "cand_2", "tie", "unresolved"]
    out = []
    for i in range(n_pairs):
        t1 = f"SAMPLE DATA answer one for synthetic question {i}. " + "detail " * (i % 5)
        t2 = f"SAMPLE DATA answer two for synthetic question {i}. " + "more " * (i % 7)
        agg = labels[i % len(labels)]
        counts = {
            "cand_1": int(agg == "cand_1") * 2,
            "cand_2": int(agg == "cand_2") * 2,
            "tie": int(agg == "tie"),
        }
        if agg == "unresolved":
            counts = {"cand_1": 1, "cand_2": 1, "tie": 0}
        votes = [
            {
                "annotator": f"expert_{k}",
                "vote": lab,
                "raw_winner": "model_a",
                "source_row": 1000 * i + k,
                "retained": True,
            }
            for k, lab in enumerate(lab for lab, c in counts.items() for _ in range(c))
        ]
        n1, n2 = len(t1.split()), len(t2.split())
        out.append(
            {
                "pair_id": f"sample_{i:03d}",
                "question_id": 1000 + i,
                "turn": 1,
                "category": CATEGORIES[i % len(CATEGORIES)],
                "question": f"SAMPLE DATA synthetic question {i}?",
                "candidates": {
                    "cand_1": {
                        "text": t1,
                        "sha256": sha256_text(t1),
                        "whitespace_tokens": n1,
                        "source_models": ["sample-model-x"],
                    },
                    "cand_2": {
                        "text": t2,
                        "sha256": sha256_text(t2),
                        "whitespace_tokens": n2,
                        "source_models": ["sample-model-y"],
                    },
                },
                "longer_candidate": "equal" if n1 == n2 else ("cand_1" if n1 > n2 else "cand_2"),
                "human": {
                    "aggregate": agg,
                    "counts": counts,
                    "n_votes_source": len(votes),
                    "n_votes_retained": len(votes),
                    "votes": votes,
                },
                "source_rows": [v["source_row"] for v in votes],
                "manifest_index": i,
            }
        )
    return out


class FakeJudge:
    """Deterministic fake judge for tests and SAMPLE DATA demos.

    `script` maps call index -> ChatResult or raw content; default behaviour prefers the
    first displayed answer most of the time and is invalid on every 11th call.
    """

    def __init__(self, script: dict[int, ChatResult | str] | None = None) -> None:
        self.calls: list[str] = []
        self.script = script or {}

    def chat(self, prompt: str) -> ChatResult:
        i = len(self.calls)
        self.calls.append(prompt)
        scripted = self.script.get(i)
        if isinstance(scripted, ChatResult):
            return scripted
        if isinstance(scripted, str):
            content = scripted
        elif i % 11 == 10:
            content = "not json"
        else:
            h = int(sha256_text(prompt)[:8], 16) % 10
            verdict = "A" if h < 6 else ("B" if h < 9 else "tie")
            content = f'{{"verdict": "{verdict}", "rationale": "SAMPLE DATA rationale."}}'
        return ChatResult(
            "ok",
            content,
            0.5 + (i % 7) * 0.1,
            "stop",
            400 + i,
            20,
            {
                "total_duration": 1,
                "load_duration": 0,
                "prompt_eval_duration": 1,
                "eval_duration": 1,
                "model": "fake",
            },
        )


def sample_freeze(manifest_sha: str) -> dict[str, Any]:
    return {
        "study_hash": "SAMPLE" + "0" * 58,
        "config_sha256": "SAMPLE",
        "prompt_sha256": "SAMPLE",
        "prompt_version": "judge-prompt-v1",
        "parser_version": "parser-v1",
        "parser_sha256": "SAMPLE",
        "metrics_version": "metrics-v1",
        "metrics_sha256": "SAMPLE",
        "config_version": "SAMPLE",
        "manifest_sha256": manifest_sha,
        "model_tag": "SAMPLE-fake",
        "model_digest": "SAMPLE-fake-digest",
        "ollama_version": "SAMPLE",
        "sampling_seed": 0,
        "sampling_rule_version": "SAMPLE",
        "aggregation_rule_version": "SAMPLE",
        "decoding": {"options": {"temperature": 0.0}, "note": SAMPLE_LABEL},
    }


def make_sample_run(root: Path = SAMPLE_DATA_DIR) -> Path:
    manifest = synthetic_manifest()
    manifest_path = root / "manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    (root / "README.md").write_text(
        "# SAMPLE DATA\n\nSynthetic fixture for UI demos and tests. Not real source data, not "
        "real human labels, not real model output. Regenerate with `uv run lja "
        "make-sample-data`.\n",
        encoding="utf-8",
    )
    run_dir = root / "run-SAMPLE-DATA"
    if run_dir.exists():
        for f in run_dir.iterdir():
            f.unlink()
    freeze = sample_freeze(sha256_file(manifest_path))
    run_benchmark(
        run_dir,
        manifest,
        freeze,
        {"model_digest": freeze["model_digest"], "ollama_version": freeze["ollama_version"]},
        FakeJudge(),
        num_ctx=8192,
        max_consecutive_connection_errors=3,
        run_meta_extra={"label": SAMPLE_LABEL, "hardware": {"note": SAMPLE_LABEL}},
    )
    cfg = copy.deepcopy(load_config())
    cfg["source"]["dataset"].update(
        {
            "hf_repo": SAMPLE_LABEL,
            "hf_revision": SAMPLE_LABEL,
            "license": SAMPLE_LABEL,
            "sha256": SAMPLE_LABEL,
        }
    )
    cfg["analysis"]["bootstrap_resamples"] = 200
    analyze_run(run_dir, manifest, cfg, manifest_path)
    run_error_review(run_dir, manifest, cfg)
    return run_dir
