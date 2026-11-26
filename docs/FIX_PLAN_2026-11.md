# WeightLiftAI: Fix & Improvement Plan for the Coding Agent

**From:** Luis (repo owner), via an outside review of `Luisv8181/weightliftai` @ `3e66925` (2026-11-22)
**Save this as:** `docs/FIX_PLAN_2026-11.md` and work through it in order.
**Ground rule:** This repo's value is its credibility. When in doubt, claim less. Never delete negative results; relabel and keep them.

Work on a branch (`fix/claims-integrity-and-links`), one commit per numbered item, and open a PR when Priority 0 and 1 are done.

---

## Priority 0: Integrity of claims (do this first)

### 0.1 "Astra" is not GPT-6 Astra, and the Part 1 result is not blind
- The MIT CSAIL post (Takehi, Teoh, Zhang, Khattab, 2026-11-02) ran its experiments with **OpenAI GPT-6 Astra**.
- In this repo, "Astra" is **the coding agent (Antigravity) role-playing Astra** (see `docs/EXP_010_PART1_WEIGHT_READING_REPLICATION_REPORT.md`, row "Astra (Antigravity)").
- That same agent wrote `generate_benchmark_weights.py` and `gold_answers_sealed.json` in the same workspace, so its 4/4 is **not a blind test**. The answer key was within reach.

**Do:**
1. In every report and in `CLAIM_EVIDENCE_MATRIX.md`, rename "Astra" to "Coding agent (Antigravity, model: <exact model>), role-play of the Astra protocol". Remove "Flawless", "proves", "Breakthrough" and similar wording.
2. Change verdicts C1 and C3 to **"Non-blind / invalid (agent authored the benchmark and gold answers)"**.
3. Re-run Part 1 properly using the repo's own blind harness (`weight_programming/blind_benchmark.py`, `isolated_evaluator.py`):
   - Generate the networks in a separate process or session; the model under test never sees the generator code, seeds, or gold file.
   - Use at least 20 networks, not 4. State the chance rate (25% for 4 options) and report a 95% CI.
   - Note that Qwen and Gemma scored exactly at chance (1/4) by always choosing one option. Say that plainly.

### 0.2 MATH-500 (C6) did not replicate the paper's claim
- The paper reports Qwen3.5-0.8B going from **7.9% to 48.1%**.
- This repo shows **11.4% -> 11.4%** on a different model. That shows the edit **didn't hurt** math, not that it **improved** it.
- **Do:** Relabel C6 "Not replicated (improvement); edit showed no degradation on <model>". If you attempt the real replication, use the same model and protocol as the paper.

### 0.3 Single-example results labeled as verified
- C4 (loop surgery) rests on one hanging prompt. Exp 009 rests on one conversation but is labeled "EMPIRICALLY VERIFIED".
- **Do:** Relabel these "Qualitative example (n=1)". For C4, build at least 30 loop-inducing prompts plus 30 controls, and report break-out rate with CI.
- Remove clinical-efficacy wording from Exp 009 ("clinical de-escalation", "eliminates victim-blaming"). Use "in this example the patched model's reply did X". Per `AGENT_HANDOFF.md`, this is research-only, not clinical.

### 0.4 One verdict scale everywhere
Replace "CONFIRMED & REPLICATED" with this scale, and add the columns **n**, **95% CI**, **blind? (y/n)**, and **model actually used**:
`Replicated` · `Partially replicated` · `Not replicated` · `Inconclusive (n too small)` · `Non-blind / invalid` · `Not attempted`

---

## Priority 1: Repo hygiene

### 1.1 Fix missing files in `docs/replication/CLAIM_EVIDENCE_MATRIX.md`
Confirm each likely match by reading the file, then fix the link:

| Table links to (missing) | Likely real file |
|---|---|
| `experiments/008-joint-subspace-surgery/run_joint_subspace_surgery.py` | `experiments/008-joint-subspace-surgery/joint_astra_surgery.py` |
| `experiments/008-joint-subspace-surgery/evaluate_collateral_impact.py` | `experiments/009-dialogic-coherence/test_side_effects_comprehensive.py` (results: `collateral_impact_results.json`) |
| `experiments/009-dialogic-coherence/run_dialogic_coherence.py` | `experiments/009-dialogic-coherence/run_dialogic_generation.py` |
| `.../part1_weight_reading/generate_benchmark_nets.py` | `.../part1_weight_reading/generate_benchmark_weights.py` |
| `.../part1_weight_reading/evaluate_open_llms.py` | `.../part1_weight_reading/eval_local_models_part1.py` |
| `.../part1_weight_reading/part1_reading_results.json` | `.../part1_weight_reading/open_models_part1_results.json` |
| `.../part2_weight_writing/evaluate_synthesis_policy.py` | `.../part2_weight_writing/eval_weight_writing.py` |
| `.../part2_weight_writing/part2_synthesis_results.json` | `.../part2_weight_writing/part2_weight_writing_results.json` |

If a claimed artifact really doesn't exist (for example, Astra's own Part 1 answers), say so in the table instead of linking.

### 1.2 Replace all laptop-only links
Convert all 58 `file:///C:/Users/luisv/Projects/weightliftai/...` links across 10 files to relative paths from the linking file.

### 1.3 Add a link-checker test so this never happens again
Add `tests/test_doc_links.py`. Parse every `*.md` file, fail on any `file:///` link, and fail on any relative link whose target doesn't exist. Runs in GitHub Actions.

### 1.4 Bring the docs up to date
- **README experiment table:** add rows 005-011 with honest status (using the 0.4 scale).
- **`docs/AGENT_HANDOFF.md`:** update from 2026-11-04.
- **`docs/ROLLING_PAPER.md`:** add 003-011 to the claims register, including the 003 failure and relabeled replication claims.
- **`docs/AUDIT_LEDGER.jsonl`:** append one entry per run for 003-011 (model + revision, seed, script, commit, result file).

### 1.5 Confirm CI is green
Run `pip install -e ".[core,cartography,dev]" && pytest -v` on Python 3.11 and 3.12, and fix any failures. Mark GPU/model-download tests with `@pytest.mark.gpu` and skip in CI.

---

## Priority 2: Make the strongest results stronger

1. Scale the *tigre* test set to >= 60 social and >= 60 literal sentences (Luis approves every Spanish item).
2. Add hard negatives (non-romance social *tigre*, sports/team names, mixed senses).
3. Run 3-5 seeds, report means with 95% CIs, pre-register pass/fail thresholds.
4. Generalize beyond one word to 4-5 other polysemous Dominican terms from the Life Language Library.
5. Compare against glossary lookup + prompt hint and LoRA with a gate.
6. Exp 002 follow-up: lexically balanced grief set (>= 40 per cell).
7. Replace 8-prompt collateral battery with standard benchmark subsets + CIs.

---

## Priority 0.5: Branch `feat/bonsai-27b-wsp-001b` (Exp 012-014, pushed 2026-11-26)

### B1. Confirm the 27B model is actually running correctly (blocking)
- Add a parity test comparing Ollama/llama.cpp vs `gguf_1bit_loader` on 20 prompts (top-5 next tokens and per-token loss). Check `HAL-X-AI/THX-02` vs GGUF embedded vocab. Until it passes, label every Exp 014 number "pending loader validation".

### B2. Exp 014 claims
- 5/8 vs 2/8 is directional (n=8, Fisher exact p ~ 0.31).
- Report L36 37.5% animal FPR and +8.5% drift in the summary.
- Investigate why steered prompt (25% -> 0%) produced a sidecar vector that lifted accuracy (25% -> 62.5%).
- Call the sidecar activation steering, not weight-space editing.

### B3. Exp 013b headline
- Lead with "no effect on the target (0/8); no collateral drift", not with zero drift.

### B4. Exp 012 claims
- Lead with overall comparison: 7/24 vs 16/24 for deterministic baseline, negative mean held-out gain.

### B5. Reproducibility
- Move hardcoded paths in 014 scripts to env vars with documented defaults + GGUF download instructions in README.

## Definition of done for this PR
- [ ] No verdict in the repo claims more than its n, blinding, and model identity support.
- [ ] `tests/test_doc_links.py` passes; zero `file:///` links remain.
- [ ] README, AGENT_HANDOFF, ROLLING_PAPER and AUDIT_LEDGER all reflect experiments 001-011.
- [ ] CI is green on 3.11 and 3.12.
- [ ] PR description lists every relabeled claim, its old verdict, and its new verdict.
