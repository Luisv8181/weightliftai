# Pre-Registration: Experiment 015: Dominican Sense Benchmark (DSB-1) Construction

**Status:** DRAFT (commit before Phase 1 approval gate closes; do not run any model on DSB-1 until Phase 1 gate passes)  
**Protocol Reference:** [`docs/RESEARCH_PROTOCOL.md`](../../docs/RESEARCH_PROTOCOL.md)  
**Owner / PI:** Luis Vasquez, LAPC, NCC

## 1. Hypotheses Supported by This Benchmark
- Phase 2 will test **H1** (social vs. literal accuracy gap), **H2** (middle-layer probe vs. output gap), and **H3** (*tíguere* vs. *tigre* spelling shortcut).
- Phase 3 will test **H4** (gated internal fix vs. collateral), **H5** (gated internal fix vs. glossary-in-prompt), and **H6** (cross-spelling transfer).

## 2. Target Models (for Phase 2, once Phase 1 gate passes)
- `google/gemma-3-1b-it` (exact revision hash to be locked in Exp 016 PREREG)
- `google/gemma-4-E2B-it`
- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `nvidia/AceInstruct-1.5B`
- *(Bonsai-27B excluded until B1 loader parity test passes.)*

## 3. Conditions & Controls (Phase 1 Scope)
- Phase 1 builds and validates the benchmark data, scorer, sealed frame-level split, and FLORES+ collateral loader. **Zero model inference runs in Phase 1.**

## 4. Primary Metric & Phase 1 Gate (Numeric Pass Bar)
- **Item Count:** >= 210 approved items (`60` social *tíguere*, `60` social *tigre*, `60` literal *tigre*, `30` ambiguous).
- **Native Speaker Approval:** 100% of items marked `approved_by: luis` with approval date.
- **Inter-Rater Agreement:** Second Dominican Spanish speaker independently labels a random 25% sample blind to intended label; **Cohen's kappa >= 0.70**.
- **Split Integrity:** Frame-level 40% dev / 60% test split (minimal pairs never separated); sealed test file stored outside repo; only SHA-256 hash committed.
- **Scorer & Isolation Tests:** All unit tests pass, including refusal to open sealed test file when `DSB_EVAL_KEY` is unset.

## 5. Collateral Benchmark Suite (Fixed)
- 500 Spanish sentences from FLORES+ (dev/devtest) for NLL and es->en chrF.
- Literal *tigre* accuracy from DSB-1.
- 200 fixed English sentences for NLL.

## 6. Seeds
- Split RNG seed: `20261126` (locked before split execution).
- Bootstrap CI resamples: `2000`, seed `42`.

## 7. Statistical Tests (for Phase 2+ Scoring)
- Forced-choice 3-paraphrase mean log-prob accuracy + margin with 95% bootstrap CIs over items.
- Paired McNemar's test (accuracy), paired permutation test (margins), Holm-Bonferroni correction across conditions/models.

## 8. Falsification / Stop Criteria for Phase 1
- If second-rater Cohen's kappa < 0.70, revise ambiguous frames and re-run blind agreement check before proceeding to Phase 2.
