"""parser-v1 strict output contract."""

from __future__ import annotations

import pytest

from llm_judge_audit.parser import count_sentences, parse_output


@pytest.mark.parametrize(
    ("raw", "verdict"),
    [
        ('{"verdict": "A", "rationale": "A is correct."}', "A"),
        ('  {"verdict": "B", "rationale": null}\n', "B"),
        ('{"rationale": "Both fine. Equal depth.", "verdict": "tie"}', "tie"),
        ('{"verdict": "A", "rationale": "Uses GPT-3.5 style; e.g.no split."}', "A"),
    ],
)
def test_valid(raw: str, verdict: str) -> None:
    res = parse_output(raw)
    assert res.ok and res.verdict == verdict and res.error is None


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ("", "empty_output"),
        ("A", "invalid_json"),
        ('```json\n{"verdict": "A", "rationale": null}\n```', "invalid_json"),
        ('{"verdict": "A", "rationale": null} extra', "invalid_json"),
        ('{"verdict": "a", "rationale": null}', "bad_verdict"),
        ('{"verdict": "Tie", "rationale": null}', "bad_verdict"),
        ('{"verdict": 1, "rationale": null}', "bad_verdict"),
        ('{"verdict": "A"}', "wrong_keys"),
        ('{"verdict": "A", "rationale": null, "score": 3}', "wrong_keys"),
        ('{"verdict": "A", "verdict": "B", "rationale": null}', "invalid_json"),
        ('{"verdict": "A", "rationale": ""}', "bad_rationale_type_or_empty"),
        ('{"verdict": "A", "rationale": 5}', "bad_rationale_type_or_empty"),
        ('{"verdict": "A", "rationale": "One. Two. Three."}', "rationale_too_many_sentences"),
        ('{"verdict": "A", "rationale": "' + "x" * 301 + '"}', "rationale_too_long"),
        ('["A"]', "not_an_object"),
        ('{"verdict": "A", "rationale": "truncated', "invalid_json"),
    ],
)
def test_invalid(raw: str, error: str) -> None:
    res = parse_output(raw)
    assert not res.ok and res.verdict is None
    assert res.error is not None and res.error.startswith(error)


def test_sentence_count() -> None:
    assert count_sentences("One sentence") == 1
    assert count_sentences("One. Two") == 2
    assert count_sentences("Version 3.5 is better. Clearly!") == 2
