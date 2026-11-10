# Research Validation Protocol & Claims Taxonomy

**Status:** Normative Standard for WeightLiftAI Research & Behavior Engineering  
**Version:** 1.2.0 (2026-11-07)  
**Tracking:** [Issue #10](https://github.com/Luisv8181/weightliftai/issues/10), [Issue #14 (Prompts 28–31)](https://github.com/Luisv8181/weightliftai/issues/14)

---

## 1. Core Principles

WeightLiftAI is designed as an open-weight model interpretability, evaluation, and behavior-engineering workbench. Its credibility depends on falsifiable experimental science, strict separation between plumbing and discovery, and zero tolerance for unearned claims.

Every contributor, agent, and automated pipeline must obey these rules:

1. **Pre-registration:** Confirmatory claims must be pre-registered in the append-only `experiment_registry` with frozen hypotheses, primary metrics, datasets, split hashes, and success criteria *before* viewing intervention results.
2. **Exploratory vs. Confirmatory Separation:** Exploratory analyses (e.g. 2D PCA visual sweeps) must never be presented as hypothesis confirmation. Confirmatory tests must be evaluated on held-out splits.
3. **Evidence Before Assertions:** Claims must be backed by reproducible execution logs, cryptographic artifact hashes, and verified test suites.
4. **Least-Invasive Precedence:** Direct weight modifications must only be pursued when simpler interventions (prompt/context, retrieval, activation steering, or adapters) are demonstrably insufficient.
5. **Human Approval Gates:** No persistent checkpoint mutation, hidden-test access, or artifact promotion can occur without explicit human authorization.
6. **Tamper-Evident Provenance:** All lifecycle transitions and evaluation events must be preserved in a cryptographic append-only hash chain.

---

## 2. Five-Tier Claims Taxonomy & Unified Maturity Scale

To avoid conflating mathematical derivation with cognitive or mechanistic discovery, all research statements in WeightLiftAI must be classified by **Claim Level (Levels 0–4)** and **Implementation Maturity** (`proposed_research_specification`, `implemented_prototype`, `tested_implementation`, `validated_empirical_method`, `not_implemented`):

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
│         │ Sensitivity            │ influence across contexts and future     │
│         │                        │ token horizons (t' >= t).                │
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

## 3. Current Repository Claims & Maturity Matrix (Prompt 30)

This table establishes the unified maturity classification across `README.md`, `docs/JACOBIAN_LENS_RESEARCH.md`, and the codebase:

| Capability / Module | Claim Level | Unified Maturity Status | Evidence & Explicit Limitations |
|---|---|---|---|
| **Layer Atlas** (`layer_atlas.py`) | Level 1 | `tested_implementation` | Captures residual/attention/MLP representations. Evaluates cosine similarity across variants and negative controls. *Does not prove causal necessity.* |
| **Representation Map** (`representation_map.py`) | Level 1 | `tested_implementation` | Dimensionality reduction (PCA/t-SNE/UMAP). Equipped with $T(k)$, $C(k)$, and $k$-NN Jaccard metrics. *Projections are exploratory visuals, not ground-truth metric geometry.* |
| **Jacobian Lens VJP & 4-Dim Evidence** (`jacobian_lens.py`) | Level 0 & Level 2 | `tested_implementation` | Cross-position VJP ($\partial h[L, t'] / \partial h[l, t]$ for $t' \ge t$) verified against central finite differences ($\text{rel\_err} < 0.05$). Multi-context and future-horizon aggregation (`aggregate_jacobian_corpus`). Separately reports Cosine Alignment, Sensitivity, Stability, and Intervention Effects (`evaluate_four_dimension_jacobian_evidence`). |
| **Overcomplete Sparse Nonnegative J-Space** (`docs/JACOBIAN_LENS_RESEARCH.md`) | Level 2 | `proposed_research_specification` | Full-vocabulary sparse nonnegative dictionary decomposition is specified in `docs/JACOBIAN_LENS_RESEARCH.md` (`not_implemented` as an automated dictionary learner). |
| **Experiment Registry** (`experiment_registry.py`) | Level 0 | `tested_implementation` | Tamper-evident cryptographic SHA-256 event hash chain, lifecycle state machine, preregistration parameter immutability lock, human approval gates. |
| **Patch Sandboxing & Budgets** (`sandbox.py`, `inspector.py`) | Level 0 | `tested_implementation` | Cryptographic raw byte SHA256 hashing across dtypes (`bfloat16`, `float16`), bit-exact rollback, defensive budget validation (rejects NaN/Inf/empty), tied-weight aliasing detection. |
| **Known-Defect Recovery** (`toy_defect.py`) | Level 0 / Level 1 | `tested_implementation` (Oracle Test Only) | Multi-task regression and held-out validation passes gates. *Explicit Limitation:* Oracle upper-bound test with access to clean ground-truth weights; **not** blind defect diagnosis. |
| **Sealed Blind Evaluator Authority (`WSP-001A`)** (`evaluator_authority.py`, `proposer_api.py`, `isolated_evaluator.py`) | Level 3 Benchmark | `tested_implementation` | Evaluator-authoritative SQLite registration, private evaluator randomness, pipe IPC (`--ipc-stream`/`--ipc-stdin`), pre-allocation resource limits, and Ed25519 signed receipts. (`BlindRepairBenchmark` in `blind_benchmark.py` is retained as `LEGACY_SAME_PROCESS_BASELINE_HARNESS`.) |
| **LLM vs. Gradient vs. Deterministic Blind Repair (`WSP-001B`)** (`llm_proposer.py`, `ablation_runner.py`) | Level 3 | `tested_implementation` | Evaluates real LLM-generated `WeightPatch` proposals across 5 diagnostic conditions and 4 defect families against gradient and deterministic baselines. |
| **Representation-Intervention Integration** (`representation_intervention.py`) | Level 3 | `tested_implementation` | Traceable failure $\to$ circuit hypothesis $\to$ activation steering/ablation vs weight edit $\to$ counterfactual and held-out evaluation $\to$ least-invasive justification. |
| **Dominican Spanish Pilot** (`dominican_pilot_50.json`) | Level 1 | `implemented_prototype` | 50 matched triads with negative controls (`is_synthetic=True`, `has_human_review=False`). |

---

## 4. Pre-Registration Standard

Before any confirmatory experiment is executed, a manifest must be registered via `weightliftai experiment init` and transitioned to `preregistered` with:

1. **Hypothesis:** Exact directional statement.
2. **Model Identity:** Canonical HF model ID and immutable commit hash.
3. **Dataset & Splits:** Fixed dataset ID, split assignment (`test` for confirmatory), and cryptographic SHA256 of the dataset file.
4. **Primary Metric & Criterion:** Unambiguous mathematical metric and numeric passing threshold.
5. **Regression Suite:** Pre-registered list of untouched benchmarks and maximum permissible regression degradation.
6. **Hardware & Environment:** Operating system, PyTorch version, CUDA version, GPU model, and package dependencies.

---

## 5. Stage Gates for Model Promotion

No patch or intervention may be approved for model promotion unless all four gates are satisfied:

1. **Gate 1 (Integrity & Budget):** Delta contains no non-finite values, strictly adheres to tensor allowlists, and does not exceed L2 or relative norm budgets.
2. **Gate 2 (Target Gain):** Statistically significant improvement on held-out test data ($p < 0.01$ or pre-registered threshold).
3. **Gate 3 (Distribution Shift):** Confirmed generalization under shifted input distributions or paraphrased evaluation sets.
4. **Gate 4 (Untouched Capabilities & Human Sign-off):** Degradation on independent regression benchmarks must not exceed the pre-registered tolerance. Final promotion requires an explicit human operator approval reference.

---

## 6. Research Stage Grading Rubric

Every study is evaluated by the automated `StageGrader` across six mandatory stages:

| Stage | Verification Criterion | Failure Condition |
| :--- | :--- | :--- |
| **1. Literature-Grounded Hypothesis** | Citations to peer-reviewed mechanistic literature; operational failure definition. | Unsubstantiated claims, vague objectives. |
| **2. Preregistration** | Immutable `manifest_sha256`; frozen dataset hashes and primary metric criteria. | Modification of criteria post-baseline run. |
| **3. Implementation Correctness** | Central finite-difference derivative checks passing within tolerance ($\text{rel\_err} < 0.05$); unit tests passing. | Analytical gradient divergence or shape mismatch. |
| **4. Execution** | Comparative baseline ladder executed (`no_op`, random norm-matched); negative controls reported. | Testing in isolation without controls. |
| **5. Analysis** | Effect sizes, uncertainty intervals, and explicit reporting of failed runs / null results. | Cherry-picked metrics, hidden failures. |
| **6. Claim Faithfulness** | Strict alignment with Claims Taxonomy (e.g., no causal assertions without Level 3 intervention). | Conflating probe decodability with mechanism. |
