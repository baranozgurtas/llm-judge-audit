"""The dashboard reads artifacts only: it must not import inference code, and must render."""

from __future__ import annotations

import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest

from llm_judge_audit.config import PROJECT_ROOT
from llm_judge_audit.ui_data import SAMPLE_SOURCE

APP = PROJECT_ROOT / "app" / "streamlit_app.py"
FORBIDDEN = {
    "llm_judge_audit.runner",
    "llm_judge_audit.ollama_client",
    "llm_judge_audit.preflight",
    "llm_judge_audit.sample_data",
    "urllib",
    "requests",
}


def _imports(path: Path) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_app_and_loader_do_not_import_inference_code() -> None:
    for path in (APP, PROJECT_ROOT / "src" / "llm_judge_audit" / "ui_data.py"):
        assert not (_imports(path) & FORBIDDEN), path


def test_every_view_renders_with_sample_data() -> None:
    views = [
        "Overview",
        "Agreement & Coverage",
        "Order & Verbosity",
        "Pair Explorer",
        "Run Integrity & Provenance",
    ]
    for view in views:
        at = AppTest.from_file(str(APP), default_timeout=30).run()
        at.sidebar.selectbox[0].set_value(SAMPLE_SOURCE).run()
        at.sidebar.radio[0].set_value(view).run()
        assert not at.exception, (view, at.exception)
        assert any("SAMPLE DATA" in w.value for w in at.warning)


def test_overview_renders_for_default_source() -> None:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception


def test_order_view_shows_labelled_verbosity_statistics_for_real_run() -> None:
    from llm_judge_audit.config import RUNS_DIR

    if not (RUNS_DIR / "bench-4e35d65ba747" / "supplementary").exists():
        return
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.sidebar.selectbox[0].set_value("bench-4e35d65ba747").run()
    at.sidebar.radio[0].set_value("Order & Verbosity").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Paired difference (registered)" in text and "204/299" in text
    assert "Raw headline difference (supplementary, unpaired)" in text
