# Research Validation Protocol & Claims Taxonomy

**Status:** Normative Standard for WeightLiftAI Research & Behavior Engineering  
**Version:** 1.0.0 (2026-11-05)  
**Tracking:** [Issue #10](https://github.com/Luisv8181/weightliftai/issues/10)

---

## 1. Core Principles

WeightLiftAI is designed as an open-weight model interpretability, evaluation, and behavior-engineering workbench. Its credibility depends on falsifiable experimental science, strict separation between plumbing and discovery, and zero tolerance for unearned claims.

Every contributor, agent, and automated pipeline must obey these rules:

1. **Pre-registration:** Confirmatory claims must be pre-registered in the append-only `experiment_registry` with frozen hypotheses, primary metrics, datasets, split hashes, and success criteria *before* viewing intervention results.
2. **Exploratory vs. Confirmatory Separation:** Exploratory analyses (e.g. 2D PCA visual sweeps) must never be presented as hypothesis confirmation. Confirmatory tests must be evaluated on held-out splits.
3. **Evidence Before Assertions:** Claims must be backed by reproducible execution logs, cryptographic artifact hashes, and verified test suites.
4. **Least-Invasive Precedence:** Direct weight modifications must only be pursued when simpler interventions (prompt/context, retrieval, activation steering, or adapters) are demonstrably insufficient.
5. **Human Approval Gates:** No persistent checkpoint mutation or artifact promotion can occur without explicit human authorization.

---

## 2. Five-Tier Claims Taxonomy

To avoid conflating mathematical derivation with cognitive or mechanistic discovery, all research statements in WeightLiftAI must be classified into one of five rigorous tiers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLAIMS TAXONOMY                                  │
├─────────┬────────────────────────┬──────────────────────────────────────────┤
│ Level   │ Category               │ Operational Requirement                  │
├─────────┼────────────────────────┼──────────────────────────────────────────┤
│ Level 0 │ Mathematical / Code    │ Unit tests, gradient checks vs. central  │
│         │ Correctness            │ finite differences (rel err < 0.05),     │
│         │                        │ bit-exact rollback hash verification.    │
├─────────┼────────────────────────┼──────────────────────────────────────────┤
│ Level 1 │ Associational          │ Linear probe accuracy, cosine similarity │
│         │ Decodability           │ above null negative-control baselines    │
│         │                        │ on held-out splits with reported CIs.    │
├─────────┼────────────────────────┼──────────────────────────────────────────┤
│ Level 2 │ Directional / Output   │ Vector-Jacobian Product (VJP) output     │
│         │ Sensitivity            │ influence, logit sensitivity across a    │
│         │                        │ frozen prompt corpus with known bounds.  │
├─────────┼────────────────────────┼──────────────────────────────────────────┤
│ Level 3 │ Causal Necessity /     │ Interchange activation patching, zero-   │
│         │ Sufficiency            │ clamping, or weight ablations showing    │
│         │                        │ specific behavioral delta on 2x2 splits. │
├─────────┼────────────────────────┼──────────────────────────────────────────┤
│ Level 4 │ Robust Behavioral      │ Generalization to out-of-distribution    │
│         │ Generalization         │ (OOD) tasks with verified regression     │
│         │                        │ tolerance on untouched capabilities.     │
└─────────┴────────────────────────┴──────────────────────────────────────────┘
```

> [!WARNING]
> **Prohibited Equivalences:**
> - Level 0 (derivative passes finite differences) **$\ne$** Level 2 (reveals true output-steering direction).
> - Level 1 (representation can be decoded by a probe) **$\ne$** Level 3 (representation causally drives generation).
> - Level 3 (toy intervention fixes in-domain case) **$\ne$** Level 4 (generalizable weight-space programming).

---

## 3. Current Repository Claims & Evidence Matrix

This table establishes the current verified status of every capability in WeightLiftAI:

| Capability / Module | Claim Level | Current Codebase Status | Evidence / Limitations |
|---|---|---|---|
| **Layer Atlas** (`layer_atlas.py`) | Level 1 | Implemented & Tested | Captures residual/attention/MLP representations. Evaluates cosine similarity across variants and negative controls. *Does not prove causal necessity.* |
| **Representation Map** (`representation_map.py`) | Level 1 | Implemented & Tested | Dimensionality reduction (PCA/t-SNE). Equipped with $T(k)$, $C(k)$, and $k$-NN Jaccard metrics. *Projections are exploratory visuals, not ground-truth metric geometry.* |
| **Jacobian Lens** (`jacobian_lens.py`) | Level 0 & Level 2 Prototype | Implemented & Tested (Prototype) | Autograd VJP verified against finite differences ($\text{rel\_err} < 0.05$). *Limitations:* Shortlist bias (candidate tokens selected from Logit Lens top-$k$); normalization layer bypassed; single-prompt influence must be averaged over contexts. |
| **Experiment Registry** (`experiment_registry.py`) | Level 0 | Implemented & Tested | Append-only event log, lifecycle state machine, human approval gates, SHA256 provenance hashes. |
| **Patch Sandboxing & Budgets** (`sandbox.py`, `inspector.py`) | Level 0 | Implemented & Tested | Bit-exact parameter rollback, budget enforcement (tensors, L2 norm, relative norm, allowlists). |
| **Known-Defect Recovery** (`toy_defect.py`) | Level 0 / Level 1 | Oracle Test Only | Multi-task regression and held-out validation passes gates. *Explicit Limitation:* Oracle upper-bound test with access to clean ground-truth weights; **not** blind defect diagnosis. |
| **Dominican Spanish Pilot** (`dominican_pilot_50.json`) | Level 1 | Provisional Curriculum | 50 matched triads with negative controls. *Explicit Limitation:* Hand-curated provisional curriculum; **not** an IRB-governed, multi-speaker field recorded corpus. |

---

## 4. Pre-Registration Standard

Before any confirmatory experiment is executed, a manifest must be registered via `weightliftai experiment init` and transitioned to `preregistered` with:

1. **Hypothesis:** Exact directional statement (e.g., *"Patching Layer 13 MLP delta reduces Dominican idiom translation error by $\ge 15\%$ on held-out test split"*).
2. **Model Identity:** Canonical HF model ID and immutable commit hash.
3. **Dataset & Splits:** Fixed dataset ID, split assignment (`test` for confirmatory), and cryptographic SHA256 of the dataset file.
4. **Primary Metric & Criterion:** Unambiguous mathematical metric (e.g. `accuracy`, `mse_loss`) and numeric passing threshold.
5. **Regression Suite:** Pre-registered list of untouched benchmarks and maximum permissible regression degradation (e.g. $\le 2.0\%$).
6. **Hardware & Environment:** Operating system, PyTorch version, CUDA version, GPU model, and package dependencies.

---

## 5. Stage Gates for Model Promotion

No patch or intervention may be approved for model promotion unless all four gates are satisfied:

```
[ Gate 1: Budget & Sanity ] ──▶ [ Gate 2: Target Gain ] ──▶ [ Gate 3: OOD & Shift ] ──▶ [ Gate 4: Regression & Approval ]
  • No NaN / Inf deltas           • Held-out split test         • Distribution shift           • Regression drop <= tolerance
  • Delta norm <= budget          • Statistical significance     • Paraphrase / boundary        • Human operator sign-off
  • Exact base hash match         • p < 0.01 vs. baseline        • Robustness intact            • Signed promotion record
```

1. **Gate 1 (Integrity & Budget):** Delta contains no non-finite values, strictly adheres to tensor allowlists, and does not exceed L2 or relative norm budgets.
2. **Gate 2 (Target Gain):** Statistically significant improvement on held-out test data ($p < 0.01$ or pre-registered threshold).
3. **Gate 3 (Distribution Shift):** Confirmed generalization under shifted input distributions or paraphrased evaluation sets.
4. **Gate 4 (Untouched Capabilities & Human Sign-off):** Degradation on independent regression benchmarks must not exceed the pre-registered tolerance. Final promotion requires an explicit human operator approval reference.
