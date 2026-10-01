# Data

`raw/` holds the pinned source files exactly as downloaded (verified by SHA-256 in
`config/study.json`). `derived/` is produced only by `uv run lja rebuild-data` and is byte-identical
across rebuilds (`uv run lja rebuild-data --check`).

## Sources

| File | Source | Revision | License |
|---|---|---|---|
| `raw/mt_bench_human_judgments_human.parquet` | `lmsys/mt_bench_human_judgments`, split `human` | `f7d2896d2cc5d80f8b55c2bbc722613555233c25` | CC-BY-4.0 |
| `raw/mt_bench_question.jsonl` | `lm-sys/FastChat` `fastchat/llm_judge/data/mt_bench/question.jsonl` | `b494d0c6b4e7935f1764f8439e75da3e66beccc7` | Apache-2.0 |

The GPT-4-judgment split (`gpt4_pair`) is never downloaded or used. Categories are not a field of
the human split; they are joined from MT-bench `question.jsonl` by `question_id`, and every
conversation's user turns are checked to equal that question's text exactly.

## Observed schema of the human split (validated in `src/llm_judge_audit/data/schema.py`)

| Column | Type | Observed semantics |
|---|---|---|
| `question_id` | int64 | MT-bench question id (81–160) |
| `model_a`, `model_b` | string | source models of the two conversations (never shown to the judge) |
| `winner` | string | `model_a`, `model_b`, or `tie` (3,355 rows: 1,293 / 1,282 / 780) |
| `judge` | string | pseudonymous human annotator id, `expert_N` or `author_N` (65 ids) |
| `conversation_a`, `conversation_b` | list<struct<content, role>> | always `[user, assistant, user, assistant]` |
| `turn` | int64 | the judged turn, 1 (1,689 rows) or 2 (1,666 rows) |

Any other column set, type, winner value, turn value, annotator-id format, role sequence, or
user-turn text causes a hard `SchemaError`.

## Derived artifacts

- `pairs.jsonl` — every eligible canonical turn-1 pair (candidates deduplicated by exact text,
  `cand_1` = text with the lexicographically smaller SHA-256), with all human votes.
- `manifest.jsonl` — the 200 sampled pairs (exact content, source rows/models, human aggregate,
  frozen whitespace-token lengths).
- `manifest_meta.json` — seed, rule versions, source revisions, checksums, per-category counts.
- `audit.json` — aggregation and sampling rule text, totals, and every row exclusion, pair
  exclusion, and vote adjustment.
- `dataset_license.json`, `ATTRIBUTION.md` — license evidence and attribution.
