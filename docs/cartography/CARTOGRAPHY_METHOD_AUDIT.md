# Cartography & Jacobian Lens Method Audit

**Document:** `docs/cartography/CARTOGRAPHY_METHOD_AUDIT.md`  
**Tracking:** [Issue #14 (Prompts 28–31)](https://github.com/Luisv8181/weightliftai/issues/14)  
**Status:** Canonical Methodological Audit & Implementation Reference  
**Author:** Astra (Cognitive Meta-Architect / Weight-Space Programmer)  
**Date:** November 8, 2026  

---

## 1. Executive Summary & Review Background

During the November 6, 2026 review of WeightLiftAI's representation cartography and Jacobian Lens prototypes, reviewers identified key areas where initial code relied on simplified approximations rather than the published methodological specifications:

1. **Single-Prompt & Same-Position Approximation (Prior Defect):**
   Early prototype code evaluated Jacobians using a single input prompt at identical source and target positions ($t' = t$). This conflated prompt-specific token artifacts with invariant output-influence directions and failed to explore how source activations at step $t$ affect downstream generation across future token horizons ($t' > t$).
2. **Conflation of Observational Alignment with Causal Necessity (Prior Defect):**
   Reports occasionally treated high cosine similarity between residual stream representations (or between Jacobian directions and unembedding vectors) as proof of semantic concept representation or causal behavioral driving.
3. **Packaging & Dependency Ambiguities (Prior Defect):**
   Optional cartography dependencies (`scipy`, `pandas`, `scikit-learn`, `pydantic`, `umap-learn`) were not clearly tiered against core requirements.
4. **Maturity Terminology Drift (Prior Defect):**
   Documentation used mixed terms ("validated method", "tested prototype", "implemented") interchangeably.

This audit document formally records the resolution of these defects across **Prompts 28, 29, 30, and 31** of Issue #14.

---

## 2. Mathematical Foundations & Published Literature

WeightLiftAI's representation cartography module builds on five foundational literature lines:

1. **Gurnee et al. (2026) — *Verbalizable Representations Form a Global Workspace in Language Models*:**
   Introduced the cross-position Jacobian Lens to probe which internal activation directions are positioned to influence subsequent verbalizations. Key requirement: Jacobians must be evaluated across distinct source and target positions ($t' \ge t$) and averaged across multiple diverse contexts to cancel context-specific artifacts.
2. **Ostrow et al. (2026) — *A Mathematical Theory of the Jacobian Lens*:**
   Provided formal conditions under which the Vector-Jacobian Product (VJP) $\mathbf{v}^T \frac{\partial \mathbf{h}[L, t']}{\partial \mathbf{h}[l, t]}$ approximates the first-order downstream logit change $\Delta \ell[t']$. Emphasized vector normalization (unit $L_2$ or RMS) and robust aggregation (trimmed mean or median) to withstand outlier gradient spikes.
3. **Nostalgebraist (2020) & Belrose et al. (2023) — *The Logit Lens & Tuned Lens*:**
   Established direct unembedding of intermediate residual activations $\mathbf{W}_U \mathbf{h}[l]$. Tuned Lens demonstrated that while raw Logit Lens is informative in late layers, intermediate layers often require affine transformations or Jacobian mappings to reveal verbalizable content.
4. **Venna & Kaski (2001) / van der Maaten & Hinton (2008) — *Preserving Neighborhood Topologies*:**
   Formulated Trustworthiness $T(k)$ and Continuity $C(k)$ to detect false neighbors and false tears introduced when projecting high-dimensional manifolds to 2D/3D coordinate displays.
5. **Takehi et al. (MIT CSAIL, Nov 2, 2026) — *Sparks of Weight-Space Programming*:**
   Demonstrated that language models can read and write neural-network parameters natively. Required clean separation between activation-space dynamics (Jacobian Lens) and parameter-space modifications (Weight-Space Programming).

---

## 3. Mathematical Requirements vs. Implementation (Prompt 28)

### 3.1 Cross-Position Derivative $\frac{\partial \mathbf{h}[L, t']}{\partial \mathbf{h}[l, t]}$ for $t' \ge t$

**Requirement:**  
Let $\mathbf{h}[l, t] \in \mathbb{R}^d$ be the residual-stream activation vector at layer $l$ and sequence position $t \in \{0, \dots, T-1\}$. In an autoregressive causal transformer, a local perturbation $\delta \mathbf{h}[l, t]$ propagates to layer $L \ge l$ only at causal positions $t' \ge t$. Perturbations at $t$ cannot influence $t' < t$.

**Implementation (`src/weightliftai/cartography/jacobian_lens.py`):**
- Function: `compute_vjp(model, input_ids, layer_idx, target_vector, source_position, target_position, future_offsets, enforce_causal_horizon=True)`
- Resolves positive and negative indexing into canonical token indices $t$ and $t'$.
- Causal check:
  ```python
  if enforce_causal_horizon and tgt_idx < src_idx:
      raise ValueError(
          f"Causal horizon violation: target_position ({tgt_idx}) < source_position ({src_idx}). "
          "Published Jacobian Lens requires t' >= t."
      )
  ```
- Uses PyTorch backward autograd on a leaf intermediate activation variable with gradient retention:
  $$\text{VJP} = \mathbf{v}_{\text{target}}^T \left( \frac{\partial \mathbf{h}[L, t']}{\partial \mathbf{h}[l, t]} \right)$$
- Verification: `check_finite_differences()` verifies autograd VJPs against two-sided finite difference approximations with relative error $< 0.05$:
  $$\frac{\partial f_k}{\partial h_i} \approx \frac{f_k(\mathbf{h} + \epsilon \mathbf{e}_i) - f_k(\mathbf{h} - \epsilon \mathbf{e}_i)}{2\epsilon}$$

### 3.2 Multi-Context Expectation & Horizon Offsets

**Requirement:**  
A single prompt produces idiosyncratic Jacobian directions dominated by local lexical and syntax constraints. The published method requires taking the expectation across a diverse corpus $\mathcal{C}$:
$$\mathbf{J}_l = \mathbb{E}_{c \in \mathcal{C}} \left[ \mathbb{E}_{t, t' \ge t} \left[ \frac{\partial \mathbf{h}_c[L, t']}{\partial \mathbf{h}_c[l, t]} \right] \right]$$

**Implementation (`src/weightliftai/cartography/jacobian_lens.py`):**
- Function: `aggregate_jacobian_corpus(model, tokenizer, corpus, layer_idx, target_token_id, source_positions, future_offsets, aggregation_mode, normalization_mode)`
- Iterates across all texts $c \in \mathcal{C}$.
- Explores combinations of source positions $t$ (e.g. `"all"` or specific indices) and future offsets $\Delta t \in \text{future\_offsets}$ where $t' = t + \Delta t < T_c$.
- Computes SHA-256 fingerprint of the evaluated corpus (`corpus_sha256`) for immutable reproducibility.
- Tracks pairwise cross-context cosine stability and standard error:
  $$\text{Stability} = \frac{1}{\binom{|\mathcal{C}|}{2}} \sum_{i < j} \cos(\mathbf{J}_{l, c_i}, \mathbf{J}_{l, c_j})$$

### 3.3 Explicit Normalization & Outlier-Resistant Aggregation

**Requirement:**  
Individual Jacobian norm magnitudes vary wildly depending on prompt length and token entropy. Raw unweighted sums can be dominated by a single saturated token.

**Implementation (`src/weightliftai/cartography/jacobian_lens.py`):**
- Function `_normalize_vector(vec, mode)`:
  - `mode="none"`: Preserves raw Euclidean magnitude.
  - `mode="l2"`: $\mathbf{v} / \|\mathbf{v}\|_2$.
  - `mode="rms"`: $\mathbf{v} / \sqrt{\frac{1}{d} \sum_{i=1}^d v_i^2}$.
- Function `_aggregate_direction_stack(stack, mode, trim_fraction=0.1)`:
  - `mode="mean"`: Standard arithmetic coordinate mean.
  - `mode="median"`: Component-wise median, completely invariant to extreme gradient spikes.
  - `mode="trimmed_mean"`: Discards top and bottom $\alpha$-fraction of values per coordinate before averaging.

---

## 4. Formal Method-to-Implementation Mapping Matrix

The mapping below is codified directly in code as `JACOBIAN_LENS_METHOD_MAPPING` (`src/weightliftai/cartography/jacobian_lens.py:29`) and verified by automated unit tests:

| Paper Requirement | Mathematical Formulation | Code Function & Module | Verification Test | Pass Criterion |
| :--- | :--- | :--- | :--- | :--- |
| **Cross-Position Jacobian** | $\frac{\partial \mathbf{h}[L, t']}{\partial \mathbf{h}[l, t]}$ for $t' \ge t$ | `compute_vjp(...)` in `jacobian_lens.py` | `test_cross_position_vjp_and_causal_horizon_enforcement` | Distinct from same-position; rejects $t' < t$ with `ValueError`. |
| **Numerical Fidelity** | $\frac{\partial f}{\partial h} \approx \frac{\Delta f}{2\epsilon}$ | `check_finite_differences(...)` in `jacobian_lens.py` | `test_check_finite_differences_matches_autograd_same_and_cross_position` | Relative error $< 0.05$ across checked dimensions. |
| **Multi-Context Expectation** | $\mathbb{E}_{c \in \mathcal{C}, t, t' \ge t} [\mathbf{J}]$ | `aggregate_jacobian_corpus(...)` in `jacobian_lens.py` | `test_multi_context_and_future_position_aggregation_distinguishes_single_prompt` | Sample count $\ge 2$; single-prompt approximation diverges ($\cos < 0.99$). |
| **Normalization & Robust Aggregation** | $L_2, \text{RMS}, \text{trimmed\_mean}, \text{median}$ | `_normalize_vector()`, `_aggregate_direction_stack()` in `jacobian_lens.py` | `test_jacobian_normalization_and_aggregation_modes` | $L_2$ norm $= 1.0$, RMS $= 1.0$; median and trimmed-mean resist $100\times$ outliers. |
| **4-Dimension Evidence Separation** | Separate Alignment, Sensitivity, Stability, Intervention | `evaluate_four_dimension_jacobian_evidence(...)` in `jacobian_lens.py` | `test_four_dimension_jacobian_evidence_separation` | Reports 4 distinct dicts with negative controls and non-causal guardrails. |
| **Original High-D Metric Cartography** | Original-space cosine + bootstrap 95% CIs | `compute_layer_metrics(...)` in `layer_atlas.py` | `test_compute_layer_metrics` | Returns `evidence_type="dimension_1_cosine_alignment"` with null baseline and guardrail. |
| **Projection Distortion Diagnostics** | $T(k), C(k), R@k$ | `compute_distortion_metrics(...)` in `representation_map.py` | `test_compute_distortion_metrics` | $0 \le T(k), C(k), R@k \le 1$; raises warning on high distortion. |

---

## 5. Four Evidence Dimensions Separation (Prompt 31)

To prevent unearned causal or semantic assertions, all cartography and Jacobian Lens analyses must report results across four strictly separated dimensions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   FOUR EVIDENCE DIMENSIONS (PROMPT 31)                      │
├────────────┬─────────────────────────────┬──────────────────────────────────┤
│ Dimension  │ Category                    │ Operational Definition & Bounds │
├────────────┼─────────────────────────────┼──────────────────────────────────┤
│ **Dim 1**  │ **Cosine Alignment**        │ Observational directional cosine │
│            │ (Observational Similarity)  │ similarity between vectors       │
│            │                             │ $\cos(\mathbf{u}, \mathbf{v}) \in [-1, 1]$.│
│            │                             │ Evaluated against norm-matched   │
│            │                             │ random and unrelated controls.   │
├────────────┼─────────────────────────────┼──────────────────────────────────┤
│ **Dim 2**  │ **Sensitivity**             │ Derivative magnitude             │
│            │ (Response Magnitude)        │ $\|\text{VJP}\|_2 = \|\mathbf{v}^T \frac{\partial \mathbf{h}}{\partial \mathbf{h}}\|_2$. │
│            │                             │ Measures how strongly outputs    │
│            │                             │ react to intermediate changes.   │
├────────────┼─────────────────────────────┼──────────────────────────────────┤
│ **Dim 3**  │ **Stability**               │ Persistence across contexts,     │
│            │ (Invariance & Generalization│ prompts, positions, seeds, and   │
│            │                             │ horizons. Pairwise context cosine│
│            │                             │ stability $\in [-1, 1]$.         │
├────────────┼─────────────────────────────┼──────────────────────────────────┤
│ **Dim 4**  │ **Intervention Effects**    │ Controlled activation steering   │
│            │ (Causal Necessity/Delta)    │ $\mathbf{h} \leftarrow \mathbf{h} + \alpha \mathbf{v}$ vs. norm-matched   │
│            │                             │ random and orthogonal sham controls│
│            │                             │ on downstream logits.            │
└────────────┴─────────────────────────────┴──────────────────────────────────┘
```

### Scientific Guardrail Rules:
- **Rule 1:** High Cosine Alignment ($\text{Dim 1} > 0.90$) does **not** prove semantic representation, stability across contexts ($\text{Dim 3}$), or causal necessity ($\text{Dim 4}$).
- **Rule 2:** Causal verbs ("steers", "drives", "governs", "causes") may **only** be used when supported by Dimension 4 controlled intervention experiments with selectivity margins exceeding both random and orthogonal sham controls.
- **Rule 3:** Sensitivity ($\text{Dim 2}$) must never be conflated with Stability ($\text{Dim 3}$). A layer can be exquisitely sensitive ($\text{high } \|\text{VJP}\|_2$) yet erratic across contexts ($\text{low stability}$).

---

## 6. Dependency & Packaging Architecture (Prompt 29)

Cartography modules depend on numerical and manifold packages that should not bloat minimal core installations. Dependencies are organized into three explicit tiers in `pyproject.toml`:

```toml
[project]
name = "weightliftai"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
core = [
    "numpy>=1.26",
    "torch>=2.2",
    "transformers>=4.45",
    "accelerate>=0.34",
    "safetensors>=0.4",
    "cryptography>=41.0",
]
cartography = [
    "numpy>=1.26",
    "torch>=2.2",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "scipy>=1.10",
    "pydantic>=2.0",
    "umap-learn>=0.5",
]
dev = [
    "pytest>=8.0",
    "ruff>=0.6",
    "numpy>=1.26",
    "torch>=2.2",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "scipy>=1.10",
    "pydantic>=2.0",
    "cryptography>=41.0",
]
```

### Import Reliability Verification:
1. `import weightliftai.cartography` succeeds cleanly when `.[cartography]` is installed.
2. In `representation_map.py`, `umap` is loaded dynamically with graceful degradation and actionable installation guidance if omitted (`"UMAP is not installed. Install with 'pip install umap-learn' or use algorithm='pca'"`).
3. `test_cartography_public_imports_and_pyproject_dependencies()` in `tests/test_jacobian_lens.py` asserts that all eight public cartography symbols are exported and all seven declared packages exist in packaging metadata.

---

## 7. Unified Maturity Classification (Prompt 30)

To eliminate misleading status descriptions across `README.md`, `docs/REPRESENTATION_CARTOGRAPHY.md`, and `docs/RESEARCH_VALIDATION_PROTOCOL.md`, all components are classified under the standardized five-tier scale:

| Tier | Status Label | Meaning |
| :---: | :--- | :--- |
| **0** | `not_implemented` | Not started or abandoned. |
| **1** | `proposed_research_specification` | Formal mathematical specification or product architecture document authored; code not yet built. |
| **2** | `implemented_prototype` | Code executes on synthetic or pilot data; lacks exhaustive tests or held-out replication. |
| **3** | `tested_implementation` | Passes unit, regression, finite difference, and sandboxing test suites; independently auditable. |
| **4** | `validated_empirical_method` | Replicated across independent model families, out-of-distribution corpora, and external benchmarks with published confidence intervals. |

### Component Inventory & Status:
- `layer_atlas.py` (`extract_layer_representations`, `compute_layer_metrics`, `generate_atlas_html`): **`tested_implementation`**
- `representation_map.py` (`project_representations`, `compute_distortion_metrics`, `generate_map_html`): **`tested_implementation`**
- `jacobian_lens.py` (`compute_vjp`, `aggregate_jacobian_corpus`, `evaluate_four_dimension_jacobian_evidence`): **`tested_implementation`**
- `schemas.py` & `validators.py` (Pydantic models, leakage validation, graph cycle checks): **`tested_implementation`**
- `dominican_pilot.py` (50 matched Spanish cultural idiom triads): **`implemented_prototype`**
- `medical_pilot.py` (50 NLM MeSH matched clinical knowledge triads): **`implemented_prototype`**
- Overcomplete Sparse Nonnegative J-Space Dictionary Learner: **`proposed_research_specification`**
- Standalone Interactive Web GUI Explorer: **`proposed_research_specification`**

---

## 8. Documented Ambiguities, Decisions & Limitations

1. **Token Horizon Handling:**
   - *Ambiguity:* The original paper states $t' \ge t$, but does not specify whether $t'$ should include the final EOS token or be restricted to a fixed forward window.
   - *Decision:* `future_offsets` is configurable (e.g. `[0, 1, 2]`), and token indices are clamped to $T_c - 1$.
2. **Context Weighting in Aggregation:**
   - *Ambiguity:* Should multi-context expectations weight each context equally, or weight by prompt token length?
   - *Decision:* Contexts are weighted equally by default in `aggregate_jacobian_corpus`, as each context represents one independent conceptual realization.
3. **Consumer Hardware VRAM Bounds:**
   - *Limitation:* Computing full Jacobian matrices $\mathbf{J} \in \mathbb{R}^{d \times d}$ requires $d$ backward passes. On consumer GPUs (e.g. RTX 5070 with 8GB VRAM), we compute Vector-Jacobian Products (VJPs) $\mathbf{v}^T \mathbf{J}$ with $O(1)$ backward passes rather than full matrix materialization. This is mathematically exact for candidate direction readouts while keeping peak memory under 4GB.
