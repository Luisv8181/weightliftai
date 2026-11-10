# Jacobian Lens and J-Space Research Module

**Status:** `tested_implementation` (Cross-Position VJP, Multi-Context Aggregation, 4-Dimension Evidence Separation) / `proposed_research_specification` (Overcomplete Sparse Nonnegative J-Space Dictionary Decomposition)

**Tracking:** [Issue #6](https://github.com/Luisv8181/weightliftai/issues/6), [Issue #14 (Prompts 28–31)](https://github.com/Luisv8181/weightliftai/issues/14)

## Unified Maturity Classification (Prompt 30)

To reconcile `README.md`, `docs/RESEARCH_VALIDATION_PROTOCOL.md`, and `src/weightliftai/cartography/`, every component is classified using a single standardized five-tier maturity scale:
- `proposed_research_specification`
- `implemented_prototype`
- `tested_implementation`
- `validated_empirical_method`
- `not_implemented`

| Component | Maturity Status | Implementation Location | Validation Evidence & Open Limitations |
| :--- | :--- | :--- | :--- |
| **Cross-Position VJP (`d h[L, t'] / d h[l, t]` for `t' >= t`)** | `tested_implementation` | `compute_vjp()`, `check_finite_differences()` in `src/weightliftai/cartography/jacobian_lens.py` | Verified against central finite differences (`rel_err < 0.05`) across both same-position (`t' = t`) and future-horizon (`t' > t`) pairs (`tests/test_jacobian_lens.py`). |
| **Multi-Context & Multi-Position Aggregation** | `tested_implementation` | `aggregate_jacobian_corpus()` in `src/weightliftai/cartography/jacobian_lens.py` | Evaluates across multiple contexts, configurable `source_positions`, `future_offsets`, `aggregation_mode` (`mean`, `median`, `trimmed_mean`), and `normalization_mode` (`none`, `l2`, `rms`). |
| **4-Dimension Evidence Separation (Alignment, Sensitivity, Stability, Intervention)** | `tested_implementation` | `evaluate_four_dimension_jacobian_evidence()` in `src/weightliftai/cartography/jacobian_lens.py` | Separately reports (1) Cosine Alignment, (2) VJP Norm Sensitivity, (3) Cross-Context/Position Stability, and (4) Controlled Activation Intervention vs. norm-matched random and orthogonal sham controls. |
| **Layer Atlas & Representation Map** | `tested_implementation` | `layer_atlas.py`, `representation_map.py` | Captures layer-wise residual/attention/MLP activations and computes `T(k)`, `C(k)`, and `k`-NN Jaccard distortion diagnostics. Visual 2D/3D projections remain exploratory aids, not literal conceptual geometry. |
| **Sparse Nonnegative J-Space Dictionary Decomposition** | `proposed_research_specification` | Specified in this document; `not_implemented` as a full vocabulary dictionary learner | Current code computes exact VJP readouts and candidate token pools rather than fitting an overcomplete sparse nonnegative dictionary over the full vocabulary. |

## Why this belongs in WeightLiftAI

WeightLiftAI investigates model behavior through parameters, activations, SAE features, and controlled interventions. A Jacobian Lens adds a complementary question: which directions in an internal activation at `(layer l, source position t)` are positioned to influence what the model could verbalize at later positions `t' >= t`?

This is distinct from weight space. Model weights determine the transformations; the Jacobian Lens analyzes how activation perturbations propagate through those transformations toward output logits.

## Research basis

Anthropic's 2026 work introduces the Jacobian lens (J-lens). For a residual-stream activation at layer `l` and source position `t`, the method studies how perturbations affect later residual states and output logits (`t' >= t`). The paper averages Jacobian effects across positions and contexts to reduce prompt-specific artifacts. The resulting token-associated directions are read out through the model's unembedding.

Primary references:
- [Anthropic research overview: A global workspace in language models](https://www.anthropic.com/research/global-workspace)
- [Gurnee et al. (2026): Verbalizable Representations Form a Global Workspace in Language Models](https://arxiv.org/abs/2607.15495)
- [Full paper and methods](https://transformer-circuits.pub/2026/workspace/index.html)
- [Ostrow et al. (2026): A Mathematical Theory of the Jacobian Lens](https://arxiv.org/abs/2610.31955)

## Mathematical sketch & Method-to-Code Mapping (Prompt 28)

Let `h[l, t]` be the residual-stream activation at layer `l` and token position `t`. A local perturbation propagates to later residual states according to the cross-position Jacobian:

`d h[L, t'] / d h[l, t]`, for `t' >= t`.

A context- and position-averaged map is estimated as:

`J_l = E_{context c, t, t' >= t} [ d h_c[L, t'] / d h_c[l, t] ]`.

In `src/weightliftai/cartography/jacobian_lens.py` (`JACOBIAN_LENS_METHOD_MAPPING`), each step is mapped directly to code:
1. **Cross-Position Derivative (`t' >= t`):** `compute_vjp(..., source_position=t, target_position=t_prime, future_offsets=..., enforce_causal_horizon=True)` computes `v^T (d h[L, t'] / d h[l, t])` and raises a `ValueError` if `t' < t`.
2. **Multi-Context & Horizon Expectation:** `aggregate_jacobian_corpus(..., source_positions="all", future_offsets=[0, 1], aggregation_mode="mean"|"median"|"trimmed_mean", normalization_mode="none"|"l2"|"rms")` aggregates across all valid `(context, t, t')` triples and reports pairwise cross-context cosine stability and standard error.
3. **Four-Dimension Evidence Separation (Prompt 31):** `evaluate_four_dimension_jacobian_evidence(...)` separates:
   - `dimension_1_cosine_alignment`: observational directional cosine similarity vs. norm-matched random and unrelated-token negative controls.
   - `dimension_2_sensitivity`: distribution of `||VJP||_2` across contexts and positions.
   - `dimension_3_stability`: cross-context pairwise cosine mean and current-vs-future horizon persistence.
   - `dimension_4_intervention_effects`: reversible forward-hook activation steering at `(layer_idx, t)` compared against norm-matched random and Gram-Schmidt orthogonal sham directions.

## Experiment registry requirements

Every run should record:
- Model and tokenizer IDs plus immutable revisions.
- Architecture adapter and exact residual tensor definition.
- Prompt-corpus ID, version, split, license/provenance, and content hash (`corpus_sha256`).
- Source layer, source token positions (`source_positions`), future-output horizon (`future_offsets`), and aggregation/normalization rules.
- Baseline, sham, random-control, intervention, and regression metrics.

## Non-claims and safeguards

- J-space is not weight space and is not a separate physical model layer.
- A J-lens readout is not a complete transcript of hidden reasoning.
- High cosine alignment (`dimension_1_cosine_alignment`) is not proof of semantic concept representation, cross-context stability (`dimension_3_stability`), or causal necessity (`dimension_4_intervention_effects`).
- Do not assume findings transfer unchanged across model families without empirical verification.

## Delivery sequence

- [x] Add architecture adapter contract and tiny-model numerical finite-difference tests (`tests/test_jacobian_lens.py`).
- [x] Implement cross-position (`t' >= t`) and multi-context Jacobian readout (`jacobian_lens.py`).
- [x] Add context aggregation, normalization options, and reproducibility metadata (`corpus_sha256`).
- [x] Add logit-lens comparison and random/orthogonal-sham controls (`evaluate_four_dimension_jacobian_evidence`).
- [ ] Add overcomplete sparse nonnegative dictionary extraction (`proposed_research_specification`).
