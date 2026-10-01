# LLM Judge Audit

**Question.** On N = 200 MT-bench pairwise comparisons with expert human preferences, how often
does one local LLM judge (Ollama `gemma3:4b`) agree with the human preference? How often does
swapping the answer positions change its mapped-back decision? Does it favour the first-shown
answer or the longer one? And how reliable and costly is running it?

- **Sample:** N = 200 unique turn-1 pairs, 25 from each of the 8 MT-bench categories,
  seeded (20261001) and stratified.
- **Calls:** 400 expected judge calls: each pair is judged once in original order and once
  with the answers swapped.
- **Data:** [`lmsys/mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments),
  `human` split only, revision `f7d2896d2cc5d80f8b55c2bbc722613555233c25`,
  license **CC-BY-4.0** (Zheng et al., 2023, arXiv:2306.05685). Question categories come from
  MT-bench `question.jsonl` in `lm-sys/FastChat` @ `b494d0c` (Apache-2.0). See
  `data/derived/ATTRIBUTION.md`.
- **Judge:** a single local Gemma model, `gemma3:4b`, digest
  `a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`, Ollama 0.24.0,
  temperature 0, with no hosted APIs and no other models.
- **Scope:** exploratory. One small judge, one prompt, one dataset. A negative result is a
  valid outcome. Nothing here generalises to other judges, prompts or data.
- **Human labels are noisy.** Each pair has few annotators, and ties and disagreement are
  common. Ties and unresolved pairs are reported separately and never scored.
- **Possible contamination:** MT-bench questions and answers have been public since 2023 and
  may be in Gemma's training data.

## Current status

**NOT STARTED.** No benchmark cells have been run (0/400). The offline system, data pipeline
and frozen preflight are in place; benchmark inference waits for explicit approval.
This section is updated from the saved artifacts after any run. A run is only called
complete when all 400 cells are valid and integrity checks pass.

## Commands

```bash
uv run pytest                              # offline test suite (no network, no model)
uv run lja rebuild-data                    # rebuild the 200-pair sample from pinned raw data
uv run streamlit run app/streamlit_app.py  # launch the read-only dashboard
```

Other commands (`uv run lja --help`):

| Command | Purpose |
|---|---|
| `lja download` | fetch pinned raw files and verify their SHA-256 (already committed in `data/raw/`) |
| `lja rebuild-data --check` | verify the derived files are byte-identical to a fresh rebuild |
| `lja freeze` / `lja verify-freeze` | freeze or verify the prompt/parser/metrics/config/manifest/model hashes |
| `lja probe` | S3: at most two synthetic probe calls (stored in `results/preflight/`) |
| `lja preflight` | S4: compact PASS/FAIL report (runs tests, checks live model digest) |
| `lja run --approved` | run or resume the 400 cells (refuses without approval and a PASS preflight) |
| `lja integrity` / `lja analyze` / `lja error-review` | validate, compute metrics/provenance, select error cases |
| `lja make-sample-data` | regenerate the synthetic **SAMPLE DATA** demo run |

## Method

**Data construction** (`src/llm_judge_audit/data/`):
1. The real schema is validated strictly. Any unexpected column, type, winner value, turn,
   annotator-id format, role sequence or question-text mismatch causes a hard failure. See
   `data/README.md`.
2. Only turn-1 rows are used; turn-2 rows are logged as exclusions.
3. Candidates are deduplicated by exact text. `cand_1` is the text with the smaller SHA-256,
   which gives a pseudo-random, content-addressed original order.
4. Pairs with identical candidate texts are excluded and logged.
5. Human votes are aggregated with **aggregation-v1**:
   - Repeat identical votes from one annotator count once.
   - Conflicting votes from one annotator are all dropped.
   - The label with a strict plurality wins. Without one, the pair is `unresolved`.
   - A tie stays a tie.
   - Every adjustment is logged in `data/derived/audit.json`.
6. Results on the real data:
   - 3,355 rows; 1,666 turn-2 rows and 17 identical-text rows excluded.
   - 881 eligible pairs; 3 duplicate votes dropped.
   - Sample: 69 + 85 resolved, 32 human ties, 14 unresolved.

**Judging.**
- **Prompt:** `prompts/judge_prompt_v1.txt`. It contains only the question and the two
  responses, verbatim. No labels, annotator ids, model names or categories are included;
  tests prove this for all 400 prompts.
- **Output:** Ollama JSON-schema constrained output, then independent strict validation
  (**parser-v1**). The response must be exactly `{"verdict": "A"|"B"|"tie", "rationale":
  ≤2 sentences, ≤300 chars | null}`.
- **Failures:** invalid JSON, schema violations, truncation (`done_reason=length`),
  runtime errors and timeouts are recorded as terminal failures, never as verdicts. There
  are no retries.

**Runner.**
- One append-only, fsync'd JSONL record per `(pair_id, order)`.
- Resume skips recorded cells and refuses any mismatch in study hash, model digest, Ollama
  version or integrity.
- The integrity check detects missing, duplicate, corrupt (hash-chained per record),
  mismatched and out-of-manifest cells.

**Registered metrics** (**metrics-v1**, `src/llm_judge_audit/metrics.py`, frozen before
inference):
- Every rate is shown as numerator/denominator with a Wilson 95% interval.
- Kappa and paired differences use a pair-cluster bootstrap (seed 7331, 2,000 resamples),
  which keeps both orders of a pair in the same resample.
- **Main accuracy** = correct valid cells / valid cells on resolved pairs.
  **Coverage** = valid cells / (2 × resolved pairs). Both are also reported per order.
- **Order inconsistency** = pairs whose mapped verdict changed / pairs with two valid cells.
- **Position choice** (first / second / tie) is reported per displayed order.
- **Consistent-pair accuracy and coverage** are shown beside main accuracy and coverage.
- **Verbosity:** longer-answer selection by the judge and by humans on unequal-length pairs.
  Length is the whitespace-token count, frozen before inference. This is an association,
  not a causal finding.
- **Baseline:** a leave-one-pair-out majority over the sample's resolved human labels, with
  ties predicting `cand_1`. It is descriptive context only. Because `cand_1`/`cand_2` is
  hash-assigned, it is near chance by design. With perfectly balanced labels,
  leave-one-out majority is always wrong.
- **Error review:** up to 20 incorrect resolved pairs, selected with seed 4242 only after
  metrics are saved.

## Repository layout

```
config/study.json         frozen study configuration (sources, seeds, model, decoding)
data/raw/                 pinned source files (checksummed)
data/derived/             manifest, pairs, audit, license/attribution (deterministic)
prompts/                  versioned judge prompt
src/llm_judge_audit/      pipeline, runner, metrics, CLI
app/streamlit_app.py      read-only dashboard
results/freeze/           frozen study hashes
results/preflight/        synthetic probe outputs and preflight report (not benchmark data)
results/runs/<run_id>/    cells.jsonl, integrity/metrics/provenance/error_review JSON
sample_data/              synthetic SAMPLE DATA demo run (not study results)
tests/                    offline tests with synthetic fixtures
```

## License

Code: Apache-2.0 (`LICENSE`). Data: CC-BY-4.0 (MT-Bench Human Judgments) and Apache-2.0
(MT-bench questions). See `data/derived/ATTRIBUTION.md`.
