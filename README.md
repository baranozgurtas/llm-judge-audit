# LLM Judge Audit

**Can a small local LLM judge stand in for human preferences — and does it give the same
answer when you swap the order of the two responses?**

A reproducible, exploratory audit of one local judge (Ollama `gemma3:4b`) on **N = 200**
expert-labelled MT-bench answer pairs. Each pair is judged twice, in original and swapped
order (**400 judge calls**). The study is pre-registered in code, and every number below has
its count, denominator and 95% interval.

![Overview of the LLM Judge Audit dashboard](docs/screenshots/overview.png)

<sub>Screenshot of the read-only Streamlit dashboard rendering the committed run
`bench-4e35d65ba747`.</sub>

## At a glance

| | |
|---|---|
| **Status** | **COMPLETE.** 400/400 valid cells, integrity checks pass (0 corrupt, duplicate, mismatched, out-of-manifest, invalid, error, timeout or missing cells) |
| **Data** | [`lmsys/mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments), `human` split, revision `f7d2896`, **CC-BY-4.0** |
| **Sample** | 200 turn-1 pairs, 25 from each of 8 MT-bench categories, seeded (20261001) and stratified |
| **Judge** | One local model: `gemma3:4b` (digest `a2af6cc3…195f5a`), Ollama 0.24.0, temperature 0, JSON-schema output, no retries |
| **Run** | 2026-10-01, 72.9 min on an Apple M1 MacBook Air (8 GB RAM); no API cost |

## Key finding: the verdict depends on answer order

**Swapping the two answers changed the judge's mapped-back verdict for 57/200 pairs
(28.5%, Wilson 95% CI 22.7–35.1%).**

- **No net first-position preference.** Pooled across both orders, the judge picked the
  first-shown answer in 194/400 calls (48.5%, Wilson 43.6–53.4%).
- **But many individual pairs flip.** In 54 of the 57 flipped pairs the judge picked the
  *same displayed slot* both times: 25 always the first answer, 29 always the second.
  - This slot breakdown is a post-hoc description computed from the saved cells. It is not a
    registered metric.
- **Consistency comes at a cost in coverage.** Restricting to pairs where both orders agree
  raises accuracy to 100/116 (86.2%, Wilson 78.8–91.3%). That covers only 116/154 resolved
  pairs (75.3%, Wilson 68.0–81.5%), so the gain partly reflects abstaining on hard pairs.
- **Qualitative review.** In the seeded error review (20 of 54 incorrect resolved pairs), 15
  cases chose the same slot in both orders. Their rationales often contradicted each other
  across orders, and in 5 maths or reasoning items the judge called a wrong answer correct.

![Pair Explorer showing a pair whose verdict flipped after swapping](docs/screenshots/pair_explorer.png)

<sub>Pair Explorer, deep-linked to `p_d43e1f57ba7a61ba`. The judge chose the displayed "B"
answer in both orders, so it endorsed f(2) = 0 in one order and f(2) = −20 in the other.</sub>

## Results

Rates are numerator/denominator with Wilson 95% intervals. Bootstrap intervals use 2,000
pair-cluster resamples (seed 7331; unit = pair ID, both orders kept together).

### Agreement with human preferences (154 pairs with a resolved human preference)

| Metric | Count | Rate | 95% CI |
|---|---|---|---|
| Accuracy, all valid decisions (both orders) | 237/308 | 76.9% | Wilson 71.9–81.3%; pair bootstrap 71.4–82.5% |
| Coverage (valid decisions / 2 × resolved pairs) | 308/308 | 100.0% | Wilson 98.8–100.0% |
| Accuracy, original order only | 120/154 | 77.9% | Wilson 70.7–83.7% |
| Accuracy, swapped order only | 117/154 | 76.0% | Wilson 68.6–82.0% |
| Judge said "tie" on a resolved pair (scored incorrect) | 3/308 | 1.0% | Wilson 0.3–2.8% |
| No-model majority baseline, leave-one-pair-out (context only) | 85/154 | 55.2% | Wilson 47.3–62.8% |

- **Cohen's κ** (human vs judge, valid resolved cells) = **0.540**, pair bootstrap
  0.422–0.644.
- **Accuracy, original minus swapped:** +1.9 pp, pair bootstrap −5.8 to +9.1 pp.
- **Not scored, reported separately:**
  - 32 human-tie pairs (the judge said tie on 2 of their 64 cells)
  - 14 unresolved pairs
  - 0 excluded annotations in the sample

### Verbosity association (196 pairs with unequal length; 151 have a resolved human label)

| Statistic | Pairs | Count | Rate | 95% CI |
|---|---|---|---|---|
| Judge chose the longer answer (decisive calls) | all 196 | 267/387 | 69.0% | Wilson 64.2–73.4%; supplementary pair bootstrap 63.7–74.2% |
| Humans chose the longer answer | 151 resolved | 102/151 | 67.5% | Wilson 59.7–74.5% |
| Judge chose the longer answer, same pairs as humans | 151 resolved | 204/299 | 68.2% | Wilson 62.7–73.2%; supplementary pair bootstrap 61.6–74.3% |

- **Paired difference (registered),** on the same 151 pairs: 68.23% − 67.55% = **+0.7 pp**.
  Pair bootstrap −5.7 to +7.1 pp.
- **Raw headline difference (supplementary, unpaired):** 68.99% − 67.55% = **+1.4 pp**. Pair
  bootstrap −4.7 to +7.9 pp.
  - The two rates cover different pair sets, so this is not a paired comparison.
- **Interpretation:** on this sample the judge's preference for longer answers is similar to
  the humans'. This is an association, not evidence that length causes either choice.

![Verbosity section of the dashboard](docs/screenshots/verbosity.png)

### Operational quality

| | |
|---|---|
| Cells | 400 expected · 400 attempted · 400 valid · 0 invalid output · 0 runtime error · 0 timeout · 0 missing |
| Latency per call | median 10.1 s · p95 19.7 s · max 31.5 s (first call, includes model load) |
| Tokens | 265,078 prompt (median 564, max 1,771) · 19,089 completion (median 47) |
| Wall clock | 72.9 min (2026-10-01 21:00:06 → 22:12:58 UTC) |
| Cost | $0 in API fees (local inference); energy and hardware time not priced |

## Dashboard

```bash
uv run streamlit run app/streamlit_app.py
```

The dashboard reads saved artifacts only; loading it never calls the model. It has five
views, each with deep links (`?view=overview|agreement|order|pairs|integrity`, and
`&pair=<pair_id>` in the Pair Explorer):

| Agreement & Coverage | Order & Verbosity |
|---|---|
| ![Agreement & Coverage view](docs/screenshots/agreement.png) | ![Order & Verbosity view](docs/screenshots/order.png) |
| **Pair Explorer** | **Run Integrity & Provenance** |
| ![Pair Explorer view](docs/screenshots/pair_explorer.png) | ![Run Integrity & Provenance view](docs/screenshots/integrity.png) |

All screenshots are real browser renders of the dashboard against the committed run. They
were captured with Chrome at 1440 px wide, at 2× device pixel ratio.

## How it works

```
 pinned raw data ──► data pipeline ──► 200-pair manifest ──► frozen study ──► runner ──► cells.jsonl
 (HF @ f7d2896,      schema checks,      seeded, stratified,    prompt/parser/    400 calls,   append-only,
  SHA-256)           dedup, vote         byte-identical         metrics/config/   no retries,  hash per record
                     aggregation         rebuilds               model hashes      resumable
                                                                                      │
 Streamlit dashboard ◄── metrics.json · integrity.json · provenance.json · error_review.json · supplementary/
 (read-only)
```

1. **Data** (`src/llm_judge_audit/data/`)
   - Downloads the pinned `human` split and MT-bench `question.jsonl`, verifying SHA-256.
   - Validates the real schema loudly and keeps turn-1 rows only.
   - Deduplicates candidates by exact text and excludes identical-text pairs.
   - Aggregates expert votes by strict plurality. Repeat votes from one annotator count once;
     conflicting votes are dropped. Ties and unresolved pairs are never turned into decisive
     labels.
   - Logs every exclusion and adjustment to `data/derived/audit.json`.
2. **Judging**
   - **Prompt:** `prompts/judge_prompt_v1.txt` contains only the question and the two answers,
     verbatim. Tests prove that no labels, annotator ids, model names or categories reach the
     judge.
   - **Swapping:** the swapped call exchanges only the answer positions, and each verdict is
     mapped back to the original answers.
   - **Output:** Ollama JSON-schema output, then strict independent validation
     (`parser-v1`). Invalid output, truncation, errors and timeouts are terminal failures,
     never verdicts.
3. **Runner and integrity**
   - One fsync'd JSONL record per `(pair_id, order)`, each carrying a record hash.
   - Resume refuses any mismatch in study hash, model digest or Ollama version.
   - The integrity check detects missing, duplicate, corrupt, mismatched and out-of-manifest
     cells.
4. **Analysis**
   - **Registered metrics** (`metrics-v1`) were frozen before inference.
   - **Error review:** a seeded selection of incorrect pairs, made after the metrics were
     saved.
   - **Supplementary analysis:** a post-hoc `supplementary-analysis-v1` that never replaces a
     registered value.

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12+. No model or network access
is needed to test, rebuild the sample, or view results.

```bash
uv sync                                    # install pinned dependencies (uv.lock)
uv run pytest                              # offline test suite (no network, no model)
uv run lja rebuild-data                    # rebuild the 200-pair sample from pinned raw data
uv run lja rebuild-data --check            # verify derived files are byte-identical
uv run streamlit run app/streamlit_app.py  # launch the read-only dashboard
```

Other commands (`uv run lja --help`):

| Command | Purpose |
|---|---|
| `lja download` | re-fetch the pinned raw files and verify SHA-256 (already committed in `data/raw/`) |
| `lja freeze` / `lja verify-freeze` | freeze, or verify, the prompt, parser, metrics, config, manifest and model hashes |
| `lja integrity` / `lja analyze` | validate the cell log; recompute metrics and provenance from saved cells |
| `lja supplementary` | post-hoc supplementary analysis from saved records |
| `lja probe` / `lja preflight` / `lja run --approved` | inference path: synthetic probes, preflight report, the 400-cell run. Requires local Ollama with the exact frozen `gemma3:4b` digest |
| `lja make-sample-data` | regenerate the synthetic **SAMPLE DATA** demo run (not study results) |

## Limitations

- **Exploratory, narrow scope.** One small local judge, one prompt, 200 pairs from one
  dataset. The results do not generalise to other judges, prompts, domains or multi-turn
  conversations.
- **Noisy human labels.**
  - 109 of the 200 pairs rest on a single retained vote.
  - 32 pairs are human ties and 14 are unresolved.
  - Agreement is measured against these labels, not against ground truth.
- **Wide intervals.** With N = 200, differences of a few points are within noise.
- **Possible training-data contamination.** MT-bench questions and answers have been public
  since 2023 and may be in Gemma's training data.
- **Determinism.** Temperature 0 does not guarantee identical outputs on rerun.
- **Interval caveat.** Registered Wilson intervals on pooled rates treat a pair's two order
  judgments as independent. Supplementary pair-clustered intervals are shown alongside them,
  and they can be wider or narrower.
- **Verbosity is an association**, not a causal estimate.

## Data attribution and license

- **MT-Bench Human Judgments**: Zheng, Chiang, Sheng, Zhuang, Wu, Zhuang, Lin, Li, Li, Xing,
  Zhang, Gonzalez, Stoica. *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena*,
  arXiv:2306.05685 (2023).
  - Source: `lmsys/mt_bench_human_judgments`, `human` split, revision
    `f7d2896d2cc5d80f8b55c2bbc722613555233c25`.
  - License: **CC-BY-4.0**.
  - The GPT-4-judgment split is not used.
- **MT-bench question categories**: `lm-sys/FastChat` @ `b494d0c`,
  `fastchat/llm_judge/data/mt_bench/question.jsonl`. License: **Apache-2.0**.
- **Changes made:**
  - turn-1 rows only
  - exact-text deduplication
  - vote aggregation (`aggregation-v1`)
  - a 200-pair stratified sample
- **Unchanged:** human labels are never modified.
- **Third-party outputs:** candidate responses were generated by third-party models and are
  used here only for non-commercial research evaluation.
- Full details: `data/derived/ATTRIBUTION.md` and `data/derived/dataset_license.json`.
- **Code license:** Apache-2.0 (`LICENSE`).

## Repository layout

```
config/study.json          frozen study configuration (sources, seeds, model, decoding)
data/raw/                  pinned source files (checksummed)
data/derived/              manifest, pairs, audit, license/attribution (deterministic)
prompts/                   versioned judge prompt
src/llm_judge_audit/       data pipeline, runner, integrity, metrics, CLI
app/streamlit_app.py       read-only dashboard
results/freeze/            frozen study hashes
results/preflight/         synthetic probe outputs and preflight report (not benchmark data)
results/runs/<run_id>/     cells.jsonl, integrity / metrics / provenance / error_review JSON,
                           supplementary/ (post-hoc, versioned)
sample_data/               synthetic SAMPLE DATA demo run (not study results)
docs/screenshots/          dashboard screenshots used in this README
tests/                     offline tests (synthetic fixtures + checks on committed artifacts)
```

## Corrections log

- **2026-10-02: verbosity reporting (documentation only).**
  - Earlier text put a "+0.7 pp paired difference" next to the 69.0% vs 67.5% headline rates
    without saying that it uses only the 151 resolved unequal pairs (judge 204/299).
  - An audit from saved records found no metric implementation defect; all registered values
    and bootstrap intervals reproduce exactly.
  - The README and dashboard now label each statistic, and report the unpaired difference
    (+1.4 pp) separately.
  - `metrics.json`, the cell log and all frozen files are unchanged.
