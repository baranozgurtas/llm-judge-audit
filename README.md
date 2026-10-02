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

**COMPLETE.** All 400/400 cells are valid: 0 invalid outputs, 0 runtime errors, 0 timeouts,
0 missing. Integrity checks pass with 0 corrupt, duplicate, mismatched or out-of-manifest
records.
- Run: `results/runs/bench-4e35d65ba747`, study hash `4e35d65b…`, executed from clean commit
  `6e72ff3`.
- Time: 2026-10-01 21:00:06 → 22:12:58 UTC (72.9 min wall clock).
- Artifacts: `metrics.json`, `integrity.json`, `provenance.json` and `error_review.json` in the
  run directory.

## Results (exploratory; this sample, judge and prompt only)

Every rate is numerator/denominator with a Wilson 95% interval. Bootstrap intervals use 2,000
pair-cluster resamples (unit = pair ID, both orders resampled together).

Rates that pool both orders count cells as units: main accuracy, coverage, judge-tie rate,
pooled position choice and judge longer-answer rate. Their registered Wilson interval treats a
pair's two judgments as independent. The post-hoc `supplementary-analysis-v1` adds pair-cluster
bootstrap intervals for these without replacing the registered ones. Clustering widens some
intervals and narrows others: for pooled first-position choice, 43.6–53.4% (Wilson) becomes
45.0–52.0% (clustered), because a pair's two order judgments are negatively correlated.

**Human agreement** (154 pairs with a resolved human preference)

| Metric | Count | Rate | 95% CI |
|---|---|---|---|
| Accuracy, all valid decisions (both orders) | 237/308 | 76.9% | 71.9–81.3% (pair bootstrap 71.4–82.5%) |
| Coverage | 308/308 | 100% | 98.8–100% |
| Accuracy, original order | 120/154 | 77.9% | 70.7–83.7% |
| Accuracy, swapped order | 117/154 | 76.0% | 68.6–82.0% |
| Judge said "tie" on a resolved pair (counted incorrect) | 3/308 | 1.0% | 0.3–2.8% |
| Majority baseline, leave-one-pair-out (descriptive only) | 85/154 | 55.2% | 47.3–62.8% |

- Cohen's κ = **0.540** (bootstrap 0.422–0.644).
- Accuracy, original minus swapped: +1.9 pp (bootstrap −5.8 to +9.1 pp).
- Reported separately: 32 human-tie pairs (the judge said tie on 2 of their 64 cells),
  14 unresolved pairs, and 0 excluded annotations in the sample.

**Order robustness**

- **The mapped-back verdict changed after swapping on 57/200 pairs (28.5%, CI 22.7–35.1%).**
  - Post-hoc descriptive breakdown, not a registered metric: 54 of those 57 picked the same
    displayed slot in both orders. 25 always picked the first answer and 29 always picked the
    second.
- Position choice (valid cells):

  | Order | First (A) | Second (B) | Tie |
  |---|---|---|---|
  | Original | 90/200 | 106/200 | 4/200 |
  | Swapped | 104/200 | 95/200 | 1/200 |
  | Pooled | 194/400 = 48.5% (CI 43.6–53.4%) | 201/400 | 5/400 |

  So there is no net preference for the first position, but the decision depends on position
  for many individual pairs.
- Order-consistent pairs:
  - Accuracy 100/116 = 86.2% (CI 78.8–91.3%), at coverage 116/154 = 75.3% (CI 68.0–81.5%).
  - Compare with 76.9% at 100% coverage for all valid decisions.
  - The higher accuracy comes partly from abstaining on 38 resolved pairs, so it is not an
    improvement by itself.

**Verbosity association** (196 unequal-length pairs; 151 of them have a resolved human label)

| Statistic | Pairs | Count | Rate | 95% CI |
|---|---|---|---|---|
| Judge chose longer, decisive cells (registered) | all 196 unequal | 267/387 | 69.0% | Wilson 64.2–73.4%; supplementary pair-cluster bootstrap 63.7–74.2% |
| Humans chose longer (registered) | 151 resolved unequal | 102/151 | 67.5% | Wilson 59.7–74.5% |
| Judge chose longer, same 151 pairs as humans | 151 resolved unequal | 204/299 | 68.2% | Wilson 62.7–73.2%; supplementary pair-cluster bootstrap 61.6–74.3% |

- **Paired difference (registered):** judge rate on the 151 resolved unequal pairs minus the
  human rate on the same pairs: 204/299 − 102/151 = 68.23% − 67.55% = **+0.7 pp**.
  - Interval: percentile pair-cluster bootstrap −5.7 to +7.1 pp (2,000 resamples of pair IDs,
    seed 7334, both orders kept with their pair).
- **Raw headline difference (supplementary, unpaired):** 267/387 − 102/151 = 68.99% − 67.55% =
  **+1.4 pp**.
  - Subtracting the rounded figures (69.0 − 67.5) gives 1.5; the exact value is 1.44.
  - The two rates cover different pair sets, so this is a contrast of headline numbers, not a
    paired comparison.
  - Supplementary pair-cluster bootstrap over the 196 unequal pairs: −4.7 to +7.9 pp.
- The gap between the two differences comes from the 45 unequal pairs with a human tie or no
  resolved label. On those, the judge chose the longer answer in 63/88 decisive cells (71.6%).
  They count toward the judge's headline rate but not the human rate.
- On this sample the judge's preference for longer answers is similar to the humans'. This is
  an association, not evidence that length causes either choice.

**Operational quality**
- Cells: 400 expected, 400 attempted, 400 valid, 0 invalid, 0 errors, 0 timeouts, 0 missing.
- Latency: median 10.1 s, p95 19.7 s, max 31.5 s (the first call, which included model load).
- Tokens: 265,078 prompt tokens (median 565 per cell, max 1,771) and 19,089 completion tokens
  (median 47).
- Cost: $0 in API fees (local inference) on an Apple M1 MacBook Air with 8 GB RAM. Energy and
  hardware time are not priced.

**Error review** (seed 4242; 20 of the 54 incorrect resolved pairs)
- 15 picked the same displayed slot in both orders. Their rationales often contradict each
  other across orders.
- In 5 math/reasoning items the judge endorsed a wrong answer as correct, e.g. f(2) = −20 and
  "first place after overtaking second".
- 3 were order-consistent preferences for the longer answer against the human majority.
- 1 disagreed with a single-vote human label, and 1 was a tie in one order.
- See `error_review.json`. No labels, prompts or metrics were changed after the review.

**Interpretation limits.**
- The 95% intervals are wide (N = 200).
- Human labels come from 1–7 retained votes per pair, and 109/200 pairs rest on a single vote.
- MT-bench may be in Gemma's training data.
- Temperature 0 is not a determinism guarantee.
- These figures describe `gemma3:4b` with this prompt on this sample only.

## Corrections log

- **2026-10-02: verbosity reporting (documentation only).**
  - Earlier text put "+0.7 pp paired difference" next to the 69.0% vs 67.5% headline rates
    without saying that it uses only the 151 resolved unequal pairs (judge 204/299).
  - An audit from saved records found no metric implementation defect. All registered values,
    and all four registered bootstrap intervals, reproduce exactly.
  - The README and dashboard now label each statistic, and report the unpaired raw difference
    (+1.4 pp) separately.
  - `metrics.json`, the cell log, prompt, parser, manifest and model configuration are unchanged.
  - Details: `results/runs/bench-4e35d65ba747/supplementary/supplementary_analysis_v1.json`.

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
| `lja supplementary` | post-hoc `supplementary-analysis-v1` from saved records (no inference) |
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
