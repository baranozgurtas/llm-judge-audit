# LLM Judge Audit

Build this project from scratch. Create the complete application, data pipeline, tests, documentation, and dashboard described below. Treat this as a new software project: define its structure and implement every required component; do not rely on pre-existing project code or artifacts.

## Objective

Build a reproducible study of one local LLM judge using human pairwise preferences. Measure:

1. agreement between the judge and human preferences;
2. how often swapping answer positions changes the judge’s mapped-back decision;
3. preference for the first displayed answer and for the longer answer;
4. the reliability and operational cost of running the judge.

This is exploratory research. A negative result is valid. Never invent results or imply broader validity than this dataset supports.

## Fixed decisions

- Project name: **LLM Judge Audit**.
- Language: Python 3.12+, `uv`, pinned dependencies and committed lockfile.
- Data: the human-judgment split of `lmsys/mt_bench_human_judgments`. Verify the dataset’s actual license, source, schema and terms from authoritative sources before downloading or using it. Do not use the GPT-4-judgment split. If lawful reuse is unclear or disallowed, stop and report the exact issue.
- Judge: local Ollama `gemma3:4b` only. At setup, verify the installed model tag and digest from the machine. Register that exact digest as this study’s new model identity. Do not use Qwen, Groq, hosted APIs, paid inference, or another model. Do not download a substitute model without user direction.
- Sample: exactly 200 eligible unique pairwise examples, deterministically sampled and stratified by the dataset’s question categories with a fixed, recorded seed. Do not silently reduce N. If the source cannot provide 200 eligible examples under the stated rules, stop before inference and report counts and the reason.
- Inference: every selected pair is judged once in original A/B order and once with the answers swapped: exactly 400 expected inference cells.
- Decoding: temperature 0; record all Ollama/model settings, including context and output limits. Temperature 0 is not a determinism guarantee.
- Output: strict JSON with `verdict` (`A`, `B`, `tie`) and a rationale of at most two concise sentences or null. Do not request or save chain-of-thought. Invalid JSON, schema violations, truncation, runtime failures and timeouts are not verdicts. No retries.
- UI: Streamlit dashboard reads result artifacts only. Loading the UI never starts inference.

## Project setup and delivery rules

- Establish a clear repository structure for source code, tests, data, result artifacts, and the Streamlit app.
- This file is the sole specification; do not create a second spec.
- Work in the provided project directory and current branch. Do not create another branch or repository. Do not push or deploy.
- Implement in small, coherent milestones. After each milestone passes its checks, create a local commit with a descriptive message. Never claim a commit exists unless `git status` and `git log` verify it.
- Do not fabricate source data, human labels, licenses, model measurements, or evaluation results.

## Data construction and audit

1. Retrieve the dataset from its authoritative public source after verifying permitted use. Pin the source revision/version and download checksums. Keep raw data separate from derived data.
2. Inspect and document the real schema and the human split. The pipeline must fail loudly if expected fields or semantics do not match; do not guess field meanings.
3. Construct canonical pair records containing question text, category, candidate texts, source IDs, and human annotation records. Deduplicate identical candidate pairs deterministically. Exclude pairs whose candidate texts are identical from the eligible sampling pool and log every exclusion with a reason.
4. Aggregate human votes with a written, deterministic rule. Detect duplicate votes from one annotator and conflicting votes by one annotator on a pair; apply the documented rule and log every affected row. A human tie remains a tie; an unresolved vote remains unresolved. Never turn either into a decisive label.
5. Sample exactly 200 unique eligible pairs with fixed seed and category stratification. Save a manifest containing the exact selected content and source references, human aggregate, category, seed, sampling-rule version, source revision, and checksums. Save a machine-readable aggregation/exclusion audit and dataset attribution/license file.
6. Add a deterministic rebuild command. Running it twice from the pinned raw source must produce byte-identical derived artifacts. Add integrity tests for totals, exclusions, vote aggregation, sample size, stratification, and hashes.
7. Human labels, annotator votes, aggregate outcomes and label-revealing metadata must never be passed to the judge. Test prompt construction to prove this.

## Frozen judge prompt and inference

- Use one versioned prompt for both orders. Include the user question and the two candidate responses verbatim; ask for a preference based on answer quality. Do not include source human labels, annotator identities, gold answers, category if it could leak labels, or evaluation metrics.
- Render the swapped call by exchanging only the displayed answer positions. Preserve an explicit mapping so each model verdict can be translated to the original candidate identities.
- Keep the prompt concise and the same across all cells. Record its SHA-256.
- Use Ollama’s JSON/schema-constrained output if supported locally, followed by strict independent Python validation. Record the exact output contract and parser version.
- Freeze prompt, parser, metric definitions, manifest, model digest, and decoding configuration before the first benchmark inference. Any later change requires a new version and a separate run; never mix cells from different versions.

## Registered metrics

Show raw counts and denominators first, then rates. Give Wilson 95% intervals for rates. Bootstrap by pair ID (both orders stay in the same resample) with a fixed seed for kappa and paired comparisons.

### Human agreement
- Main accuracy: correct / valid judge decisions on pairs with a resolved human preference.
- Coverage: valid decisions / all selected pairs with resolved human preference.
- Cohen’s kappa on resolved, valid pairs with pair-cluster bootstrap 95% interval.
- Report human ties, unresolved examples and excluded annotations separately; never include them as ordinary correct/incorrect labels.

### Order robustness
- Map both order-specific verdicts back to original candidate identities.
- Report pairs with a changed mapped-back verdict / pairs with two valid outputs, with Wilson interval.
- Report original-order and swapped-order decisions separately, then position choice (first, second, tie) by displayed position with explicit denominators.
- Report accuracy on order-consistent pairs beside accuracy on all valid resolved pairs, along with coverage for both. Do not present abstention alone as improved accuracy.

### Verbosity association
- Freeze candidate length as whitespace-token count before inference.
- Among unequal-length pairs, report how often the judge selects the longer answer and how often humans selected it, each with denominator and Wilson interval. Describe this as an association, not causal proof.

### Operational quality
- Exact counts: expected, attempted, valid, invalid output, runtime error, timeout, missing, and complete cells.
- Median and p95 latency; prompt/completion tokens if Ollama reports them; wall-clock runtime.
- Cost is zero for local inference, but disclose local hardware and resource limits where measured.

### Error review
- Only after saving integrity-checked raw outputs and primary metrics, select up to 20 incorrect resolved pairs using a predeclared fixed seed. Record selected IDs and concise evidence-based observations. Do not change labels or tune prompts after inspecting these cases.

### Simple baseline
- Include a no-model majority-class baseline computed only from the development sample’s human labels, with its rule stated before results. Report it as descriptive context, not a competing LLM judge. Do not use labels from a test example to predict that same example; use leave-one-pair-out majority or category-blind fixed majority as specified and tested.

## Runner, resume, and provenance

- One atomic append-only JSONL record per `(pair_id, order)` cell; exactly 400 expected cells.
- Each record includes run/config/prompt/manifest hashes, pair ID, order, mapped verdict, visible raw response, parser result, latency, token usage if available, timestamp, and sanitized runtime metadata. Never store secrets or hidden reasoning.
- Resume skips any already-recorded cell. Resume requires exact matches for manifest, model digest, Ollama version, prompt, parser and config hashes. Never overwrite a completed record.
- No retries. A failed cell remains terminal and is reported; it is not scored as a verdict. A run with missing or failed cells is `PARTIAL`, not complete.
- Integrity validation detects missing, duplicate, corrupt, mismatched and out-of-manifest cells. Write progress atomically so interruption cannot corrupt completed records.
- Provenance includes run ID, UTC start/end, git commit and clean/dirty state, source revision/license, raw and manifest hashes, sampling seed/rule, model tag/digest, Ollama version, prompt/parser/metrics/config hashes, decoding parameters, local hardware, and exact cell status counts.
- Keep synthetic test fixtures separate from real model results and label them `SAMPLE DATA`.

## Dashboard and README

Build a polished, accessible Streamlit UI with these views:

1. Overview: study design, completion state, headline human-agreement results and limitations.
2. Agreement & Coverage: counts, denominators, intervals, kappa and majority baseline.
3. Order & Verbosity: order inconsistency, position preference, longer-answer selection.
4. Pair Explorer: question, anonymized answers, both displayed orders, judge outputs and human label; hide annotator identity.
5. Run Integrity & Provenance: hashes, model/runtime details, exact missing/error counts.

The README begins with the exact question, N=200, 400 expected calls, dataset source/license, one local Gemma judge, exploratory scope, noisy human labels, and possible training-data contamination. Every metric shows numerator/denominator and interval where applicable. State exact completion status. Never say “complete” unless all 400 cells are valid and integrity checks pass. Include one command for offline tests, one command to rebuild the sample from pinned data, and one command to launch the UI.

## Workflow and stopping gates

### S0 — Bootstrap and source verification (no model inference)
Initialize the project structure. Verify dataset source/license/schema and local Ollama/Gemma availability. Create a concise implementation checklist. If dataset use is impermissible or the required human split is unavailable, stop with evidence.

### S1 — Data and deterministic manifest (no model inference)
Implement the downloader/checksum process, data audit, vote aggregation, fixed-seed stratified 200-pair manifest, hashes and rebuild command. Run the rebuild twice and verify byte identity. Add tests. Commit this milestone locally.

### S2 — Offline evaluation system (no model inference)
Implement prompt, parser, configuration, 400-cell scheduler, append-only resumable runner, integrity checks, metrics, baseline, tests, CLI, UI and README. Use synthetic fixtures only in tests. Run formatting, lint, strict types, the full offline test suite, data rebuild and integrity checks. Commit this milestone locally.

### S3 — Local synthetic preflight
Using synthetic text only, make at most two local Gemma probe calls to verify prompt rendering, valid JSON and runtime. Record exact settings, latency, token counts and machine details. This is not benchmark data and must be stored separately. Freeze all study hashes and report the expected 400-cell runtime as an uncertain range based on the probes and actual prompt-token distribution. If a probe fails or the estimate is impractical, stop and report; do not change N or model silently.

### S4 — Benchmark inference (explicit approval required)
Before making any of the 400 benchmark calls, show a compact preflight with dataset/license verification, sample hash and N, exact model digest/settings, prompt/parser/metrics/config hashes, estimated runtime range, test results, and PASS/FAIL. Stop and wait for the user’s explicit approval. If approved, run the full 400-cell schedule locally with resume enabled and no retries. Preserve exact partial results if interrupted.

### S5 — Analysis and delivery
After a complete integrity-checked run, compute registered metrics, perform the seeded error review, populate the dashboard from saved results, update README, rerun all checks, and commit locally. If the run is partial, report partial counts and available descriptive metrics only; do not claim benchmark completion.

At each milestone, report changed files, local commit, commands/tests and results, model-call count, completion state, and remaining risks. Do not stop for routine implementation choices. The only approval gate is S4 benchmark inference.

## Out of scope

Second judge, model ensemble, Qwen, hosted APIs, paid inference, fine-tuning, agents, RAG, embeddings, accounts, database service, extra prompt variants, changing N, changing human labels to improve scores, pushing, or deployment.
