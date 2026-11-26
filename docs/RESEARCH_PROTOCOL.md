# WeightLiftAI Research Protocol v1
### Dominican Sense Benchmark (DSB-1) and the Long-Range Research Plan

**Owner / PI:** Luis Vasquez, LAPC, NCC  
**Applies to:** every coding agent working in `Luisv8181/weightliftai`, starting with Experiment 015  
**Save as:** `docs/RESEARCH_PROTOCOL.md`, and link it from the README and `docs/AGENT_HANDOFF.md`  
**Precondition:** Priority 0 and 1 of `docs/FIX_PLAN_2026-11.md` are done, and the Bonsai loader parity test (B1) has passed or Bonsai is excluded.

---

## Part A: The research question

> **When a small open model translates Dominican Spanish, does it fail because it doesn't know the local meaning, or because it knows it internally and loses it before the output? And which fix works best for each case: a glossary, fine-tuning, or a targeted internal edit, while leaving everything else untouched?**

This is the question behind Experiments 001-004 and 013-014, asked once, cleanly, at a scale that can support a paper.

### Hypotheses (pre-registered; do not edit after Phase 2 starts)

| ID | Hypothesis | Tested in |
|---|---|---|
| H1 | Models translate social *tigre/tíguere* worse than literal *tigre*. | Phase 2 |
| H2 | Middle layers separate the two senses better than the output does (a "knows-but-loses" gap). | Phase 2 |
| H3 | Spelling is a shortcut: models do better on *tíguere* than *tigre* for the same social meaning. | Phase 2 |
| H4 | When the H2 gap exists, a gated internal fix improves social accuracy without hurting literal or general Spanish. | Phase 3 |
| H5 | A gated internal fix beats a glossary-in-prompt fix. (Null expected; a null result is a valid finding.) | Phase 3 |
| H6 | A fix learned from one spelling transfers to the other spelling. | Phase 3 |
| H7 | The method generalizes to other Dominican terms, chosen by Luis. | Phase 4 |

---

## Part B: The test: Dominican Sense Benchmark (DSB-1)

Earlier tests used 8 sentences per group, one sentence pattern per item, and let spelling give away the answer. DSB-1 fixes all three.

### B1. Item design (the 2 x 2 x 2 grid)
Every social item is written as a minimal pair: the same sentence frame with only the context swapped.

| Factor | Levels | Why |
|---|---|---|
| **Sense** | social (person) / literal (animal) | The main contrast |
| **Spelling** (social only) | *tíguere* / *tigre* | Tests H3 and H6, the shortcut found in Exp 003 |
| **Cue position** | context clue **before** the word / **after** the word | Exp 004 showed causal models can't see clues that come after the word |

Also include:
- **Ambiguous items** (about 15%): no clue to the meaning. These are exploratory and not part of pass/fail, but they show whether the model just defaults to "tiger".
- **Hard negatives:** literal *tigre* in people-heavy contexts (a zoo keeper, a sports team mascot), and social *tigre* outside romance (work, neighborhood, music).
- **Topic spread:** romance, work, family, neighborhood, money, music, school. Don't let one topic carry the social sense.

### B2. Size

| Group | Minimum items |
|---|---|
| Social *tíguere* | 60 |
| Social *tigre* | 60 |
| Literal *tigre* | 60 |
| Ambiguous | 30 |
| **Total DSB-1 core** | **210** |

**Why 60:** with 60 items, a 95% confidence interval on accuracy is roughly +/-12 points at worst. That's enough to tell 20% from 50%, which 8 items can't do.

### B3. Who writes and approves items
1. A coding agent can draft frames, but **Luis approves or rewrites every Spanish item** as a native Dominican Spanish speaker. Mark each item `approved_by: luis` with a date.
2. A **second Dominican Spanish speaker** independently labels the sense of a random 25% sample, blind to the intended label.
   - **Gate:** agreement of Cohen's kappa >= 0.70. If it's lower, revise the unclear items and re-check.
3. No real patient or client language. All sentences are synthetic. Research only, not clinical use.

### B4. Splits (locked before any model runs)
- Split by **sentence frame**, not by item, so a frame never appears in both training and test. A minimal pair always stays together.
- **dev 40% / test 60%.** Agents may only use dev to build gates, vectors and patches.
- The **test split is sealed:** store it outside the agent's workspace (for example, a private folder or an encrypted file only Luis's evaluation script can open). Commit only its SHA-256 hash. Agents never read, print or train on test items.

### B5. Scoring
- **Primary metric, forced choice:** for each item, score candidate English translations under the model.
  - Use **3 paraphrases per sense** (literal: "tiger" variants; social: for example "a smooth-talking guy", "a street-smart guy", "a player". Luis approves the glosses).
  - The item is correct if the mean score of the correct-sense paraphrases beats the wrong-sense ones. Averaging 3 paraphrases prevents a single word from deciding the result.
  - Also report the **margin** (the size of that gap).
- **Secondary metric, free translation (50 test items per condition):** generate translations. Luis plus the second rater judge "sense correct? yes/no" **blind to which condition produced them**. Report kappa.
- **Internal metric (probes):** a linear probe per layer, trained on dev and scored on test, to measure whether a layer separates the senses (H2).

### B6. "Did anything else break?" set (fixed)
- **General Spanish:** 500 sentences from FLORES+ Spanish (dev/devtest). Report loss (NLL) and es->en translation quality (chrF) before and after every fix.
- **Literal *tigre*** accuracy from DSB-1 itself.
- **General English:** 200 fixed English sentences, loss only.

---

## Part C: Rules for coding agents (every experiment)

### C1. Roles: never the same agent session for two of these
| Role | Does | Must not |
|---|---|---|
| **Builder** | Writes code, scripts, dev-split tools | See the test split; decide pass bars |
| **Runner** | Executes the pre-registered run and saves raw outputs | Change code or thresholds after starting |
| **Reviewer** | Checks results against the pre-registration | Be the same agent or session that built it |
| **Subject** (WSP track only) | The model being tested | Be the agent editing the repo |

When a frontier model is the **subject**, call it through its API inside the sealed proposer environment. The repo-editing agent is never the subject.

### C2. Experiment folder template (`experiments/0NN-short-name/`)
```
PREREG.md      # committed BEFORE any run; the git hash goes in results
DATA_CARD.md   # items, sizes, who approved, split hashes
run.py         # one command reproduces everything; no hardcoded C:/ or D:/ paths
results.json   # raw per-item outputs, every seed, every condition
RESULTS.md     # fixed sections, below
```

### C3. `PREREG.md` must state, before running
1. Hypothesis IDs being tested
2. Models with exact revision hashes
3. Conditions and controls
4. Primary metric and **numeric pass bar**
5. Collateral limits
6. Seeds (at least 3 for anything trained or random)
7. Statistical tests
8. What result would count **against** the hypothesis

### C4. Required controls (every intervention experiment)
- **No change** (baseline)
- **Random fix of equal size** (norm-matched random vector or patch)
- **Gate with a random vector** (to see whether the gate alone does anything)
- **Shuffled gate** (fires at random, at the same rate)
- **Simple fix:** glossary note in the prompt
- **Standard fix:** LoRA trained on the same dev data

### C5. Statistics
- Per-item **paired** comparisons between conditions: McNemar's test for correct/incorrect, and a paired permutation test for margins.
- 95% bootstrap confidence intervals over items, for every number in every table.
- **Holm correction** when comparing across several models or conditions.
- Report effect sizes, not only p-values. Never report "x/8"-style results as findings.

### C6. Allowed result labels (nothing else)
`Supported` · `Not supported` · `Inconclusive (underpowered)` · `Invalid (protocol violation)` · `Exploratory`

**Banned words in reports:** verified, proven, flawless, breakthrough, surgical precision, clinical (for model outputs), exact (unless it's a mathematical identity).

### C7. Ledger
Append one line to `docs/AUDIT_LEDGER.jsonl` per run, including failed and aborted runs: experiment ID, commit, prereg hash, model, revision, seed, condition, result file, label.

### C8. `RESULTS.md` sections (in this order)
1. What we tested (2 sentences)
2. Pre-registered pass bar vs measured, as a table with CIs
3. Controls table
4. Collateral table
5. Label (C6)
6. What this does NOT show
7. Deviations from the pre-registration, or "none"

---

## Part D: The phased plan (each result decides the next step)

- **Phase 0 (Clean-up):** Complete Priority 0 and 1 of `docs/FIX_PLAN_2026-11.md` and pass the Bonsai loader parity test (B1) or exclude Bonsai from Phases 2-4.
- **Phase 1 (Build DSB-1, Exp 015):** Draft frames, Luis approves all Spanish items, second rater checks 25% sample (gate: Cohen's kappa >= 0.70), lock frame-level 40/60 split with sealed test hash, build scorer and FLORES+ collateral loader.
- **Phase 2 (Diagnose, Exp 016):** Evaluate 5 small models on 8 GB GPU (`Gemma 3 1B IT`, `Gemma 4 E2B IT`, `Qwen2.5-0.5B-Instruct`, `Qwen2.5-1.5B-Instruct`, `AceInstruct-1.5B`) on H1, H2, H3 and classify each model (`Already fine`, `Knows-but-loses`, `Doesn't know`, `Partial`, `Prompt solves it`).
- **Phase 3A (Internal fixes vs simpler fixes, Exp 017):** Only for "knows-but-loses" models. Compare glossary, LoRA, ungated rank-1, self-gated rank-1, self-gated steering vector, and all C4 controls (3 seeds). Pass bar: >= +25 pp social accuracy (95% CI > 0, Holm-corrected McNemar p < 0.05), literal drop <= 5 pp, FLORES+ NLL drift <= 1% and chrF drop <= 1 pt. Test spelling transfer (H6).
- **Phase 3B (Supply the knowledge, Exp 018):** For "doesn't know" models: glossary, Life Language Library retrieval, and LoRA.
- **Phase 4 (Generalize, Exp 019-021):** Expand to 5-8 Dominican terms chosen by Luis (DSB-2), >= 2 more model families, held-out term transfer, multi-term interference, 2-3 turn dialogue context, and blind human ratings on 100 items/condition.
- **Phase 5 (Weight-space programming track, Exp 022-025):** Runs in parallel from Phase 1 onward: 40+ sealed networks for blind weight reading, wire ladder rungs 2 and 4, 30+ sealed defects/rung against rule-based and gradient baselines, and bridge to DSB.
- **Phase 6 (Write-up and release):** Freeze code/data, tag release with DOI, update `docs/ROLLING_PAPER.md`, release DSB under open license, post Phase 3A pre-registration publicly before running.

---

## Part E: First tasks for the coding agent, in order

1. Commit this file as `docs/RESEARCH_PROTOCOL.md` and link it from the README and `AGENT_HANDOFF.md`.
2. Create `experiments/015-dsb1-benchmark/` with `PREREG.md` and `DATA_CARD.md` templates (Part C2).
3. Write `dsb/frames_draft.csv` with **draft** frames covering the Part B1 grid and topics. Columns: `frame_id, sense, spelling, cue_position, topic, spanish, gloss_options, approved_by, approved_date, notes`. Leave `approved_by` empty; Luis fills it in.
4. Build the scorer (`src/weightliftai/dsb/scorer.py`): forced choice with 3 paraphrases per sense, margins and bootstrap CIs. Add unit tests.
5. Build the split tool: frame-level split, sealed test written outside the repo, hash committed. Add a test that the scorer refuses to load the sealed file unless `DSB_EVAL_KEY` is set.
6. Build the FLORES+ collateral loader.
7. **Stop.** Ask Luis to review the frames before going further. Don't run any model on DSB until Phase 1's gate is met.
