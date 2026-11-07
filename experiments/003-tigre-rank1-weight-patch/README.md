# Experiment 003 — Rank-1 weight patch vs LoRA for Dominican "tíguere"

Status: pre-registered before first run (criteria below were written before any intervention was trained).

## Question
Can a single, norm-bounded rank-1 weight edit (ΔW = u vᵀ on one MLP `down_proj`) teach a small
local model to translate Dominican slang *tíguere / tigre* ("street-smart guy / player") instead of
"tiger", **without** breaking literal animal uses or general translation — and how does it compare to
a conventional LoRA baseline and a random rank-1 control of equal norm?

This is the narrow experiment that `optimustconnectofficial` Issue #15 asks for before any claim is
made about weight-space programming for DISCORS-S.

## Setup
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, fp32, local, offline.
- Observed baseline defect (greedy): "Mi novia me dejó por un tigre." → "My girlfriend left me for a lion."
- Data (`data.py`): all sentences authored for this experiment (synthetic, no human-subject data).
  - train: 8 slang + 6 animal items
  - test (held out, different sentences): 10 slang + 8 animal items
  - regression: 8 unrelated Spanish→English pairs
- Metric per item: margin = log p(correct English) − log p(swapped English), where the two English
  candidates differ only in "player" ↔ "tiger". Win = margin > 0.
- Regression: mean per-token NLL on reference translations of unrelated sentences.

## Conditions
1. `baseline` — frozen model.
2. `lora_r4` — LoRA r=4 on q_proj/v_proj, all layers, same data/steps.
3. `rank1_patch` — trainable u, v on one layer's `down_proj`, Frobenius norm ‖u‖·‖v‖ ≤ 5% of ‖W‖_F.
   Layer chosen from {6, 12, 18} by **training loss only**. Patch folded into W after training and the
   fold is checked numerically.
4. `random_rank1` — random u, v with the same norm and layer as the learned patch (negative control).

Seeds: 0, 1, 2. Item-level bootstrap (2000 resamples) for 95% CIs on test margins.

## Pre-registered success criteria (for `rank1_patch`)
- Held-out slang win rate ≥ 0.80
- Held-out animal win rate ≥ 0.875 (no more than 1 of 8 animal items wrong)
- Regression NLL increase ≤ 5% relative to baseline
- Mean held-out slang margin CI excludes the random-rank1 control mean

## Known limitations (stated up front)
- Tiny, author-written dataset; one model; one phenomenon. A pass would show feasibility of a
  narrow behavioral edit, not general dialect preservation, privacy, or DISCORS-S suitability.
- Slang test items mostly contain social context cues; the spelling *tíguere* is a possible shortcut.
  Results are additionally reported for test items spelled *tigre* (the ambiguous case).
- Candidate-scoring metric, not human evaluation of free generations (greedy samples are logged for
  inspection).

Reproduce: `python experiments/003-tigre-rank1-weight-patch/run_experiment_003.py`
