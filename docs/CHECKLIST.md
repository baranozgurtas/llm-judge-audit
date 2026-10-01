# Implementation checklist

Tracks progress against the milestones in `CLAUDE.md` (the sole specification).

## S0 — Bootstrap and source verification
- [x] uv project, Python 3.12, pinned dependencies, lockfile
- [x] Dataset `lmsys/mt_bench_human_judgments` verified on the Hugging Face Hub API:
      revision `f7d2896d2cc5d80f8b55c2bbc722613555233c25`, license `cc-by-4.0` (card metadata),
      not gated, `human` split = 3,355 rows (GPT-4 split is never downloaded)
- [x] Question categories: not present in the dataset; taken from MT-bench `question.jsonl`
      in `lm-sys/FastChat` (Apache-2.0) at the only commit touching it, `b494d0c6…`
- [x] Real schema inspected (8 columns, 4-message conversations, winner ∈ {model_a, model_b, tie})
- [x] Ollama 0.24.0 running locally; `gemma3:4b` digest
      `a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a` registered in `config/study.json`
- [x] Hardware recorded: Apple M1 MacBook Air, 8 GB RAM, macOS 26.3.1

## S1 — Data and deterministic manifest
- [x] Downloader with pinned URLs and SHA-256 verification; raw kept in `data/raw/`
- [x] Loud schema/semantics validation
- [x] Canonical pairs, content dedup, identical-text exclusion log
- [x] Vote aggregation with duplicate/conflict handling and audit log
- [x] Seeded, category-stratified 200-pair manifest + metadata + attribution
- [x] `lja rebuild-data` twice → byte-identical; integrity tests; commit

## S2 — Offline evaluation system
- [x] Frozen prompt v1, parser v1, metrics v1, config hashing, freeze file
- [x] 400-cell scheduler, append-only resumable runner, no retries
- [x] Integrity checks, provenance, metrics, baseline, error-review selection
- [x] CLI, Streamlit UI (artifact-only), README
- [x] ruff format/lint, mypy --strict, pytest; commit

## S3 — Synthetic preflight (≤ 2 probe calls)
- [x] Study frozen: study_hash 4e35d65b…; 2 synthetic probes valid (results/preflight/)
- [x] Preflight report PASS; runtime estimate 1.5–4.1 h
## S4 — Benchmark inference (explicit user approval required)
## S5 — Analysis and delivery
