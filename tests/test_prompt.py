"""Prompt construction: verbatim content, swap mapping, and no label leakage."""

from __future__ import annotations

import inspect
import re
from typing import Any

import pytest

from llm_judge_audit.config import DATA_DERIVED
from llm_judge_audit.io_utils import read_jsonl
from llm_judge_audit.prompt import (
    DISPLAY_MAPPING,
    load_template,
    map_verdict,
    prompt_hash,
    render_cell,
    render_prompt,
)

MANIFEST = read_jsonl(DATA_DERIVED / "manifest.jsonl")


def test_render_signature_accepts_only_content() -> None:
    assert list(inspect.signature(render_prompt).parameters) == [
        "question",
        "response_a",
        "response_b",
        "template",
    ]


def test_placeholders_in_answers_are_not_resubstituted() -> None:
    out = render_prompt("Q {{RESPONSE_B}}", "A has {{QUESTION}} {x}", "B {{RESPONSE_A}}")
    assert "Q {{RESPONSE_B}}" in out
    assert "A has {{QUESTION}} {x}" in out
    assert "B {{RESPONSE_A}}" in out


def test_verbatim_and_swap_exchanges_only_positions() -> None:
    m = MANIFEST[0]
    o, s = render_cell(m, "original"), render_cell(m, "swapped")
    c1, c2 = m["candidates"]["cand_1"]["text"], m["candidates"]["cand_2"]["text"]
    assert o.prompt == render_prompt(m["question"], c1, c2)
    assert s.prompt == render_prompt(m["question"], c2, c1)
    assert m["question"] in o.prompt and c1 in o.prompt and c2 in o.prompt
    # Replacing the two answers with placeholders makes both prompts identical.
    assert o.prompt.replace(c1, "<X>").replace(c2, "<Y>") == s.prompt.replace(c2, "<X>").replace(
        c1, "<Y>"
    )


@pytest.mark.parametrize(
    ("order", "verdict", "expected"),
    [
        ("original", "A", "cand_1"),
        ("original", "B", "cand_2"),
        ("swapped", "A", "cand_2"),
        ("swapped", "B", "cand_1"),
        ("original", "tie", "tie"),
        ("swapped", "tie", "tie"),
    ],
)
def test_mapping_back(order: str, verdict: str, expected: str) -> None:
    assert map_verdict(verdict, order) == expected
    assert DISPLAY_MAPPING[order][verdict] == expected if verdict != "tie" else True


def _leak_tokens(m: dict[str, Any]) -> list[str]:
    tokens = [v["annotator"] for v in m["human"]["votes"]]
    tokens += [mod for c in m["candidates"].values() for mod in c["source_models"]]
    tokens += [
        "cand_1",
        "cand_2",
        "model_a",
        "model_b",
        "unresolved",
        "expert_",
        "author_",
        m["pair_id"],
        m["category"],
    ]
    return tokens


def test_no_label_or_identity_leakage_in_any_benchmark_prompt() -> None:
    for m in MANIFEST:
        for order in ("original", "swapped"):
            prompt = render_cell(m, order).prompt
            content = m["question"] + "".join(c["text"] for c in m["candidates"].values())
            for tok in _leak_tokens(m):
                if tok in content:
                    continue  # appears verbatim in the source text itself, not injected
                assert tok not in prompt, (m["pair_id"], order, tok)
            # Nothing outside the template + verbatim content.
            residue = prompt
            for part in (m["question"], *(c["text"] for c in m["candidates"].values())):
                residue = residue.replace(part, "", 1)
            template_text = re.sub(r"\{\{[A-Z_]+\}\}", "", load_template())
            assert residue == template_text


def test_template_has_no_label_terms_and_hash_is_stable() -> None:
    tpl = load_template().lower()
    for word in ("human", "annotator", "expert", "gpt", "claude", "vicuna", "category", "gold"):
        assert word not in tpl
    assert prompt_hash() == prompt_hash()
    assert len(prompt_hash()) == 64
