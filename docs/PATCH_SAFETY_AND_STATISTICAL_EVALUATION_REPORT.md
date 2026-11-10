# WeightLiftAI Patch Safety, Statistical Evaluation, and Benchmark Ladder Report (Phases 2–4)

**Repository:** `Luisv8181/weightliftai`  
**Tracking Issues:** [Issue #12](https://github.com/Luisv8181/weightliftai/issues/12) & [Issue #14](https://github.com/Luisv8181/weightliftai/issues/14)  
**Branch:** `feat/blind-benchmark-isolation-pra`  
**Protocol Version:** `wsp-v2.0-auditable`  
**Date:** 2026-11-09  

---

## 1. Executive Summary

This report documents the implementation and experimental verification of **Phase 2 (Fail-Closed Patch Safety)**, **Phase 3 (Statistical Evaluation and Multi-Seed Benchmarking)**, and **Phase 4 (Multi-Architecture Benchmark Ladder)** for WeightLiftAI's Weight-Space Programming subsystem.

### Key Milestones Achieved:
1. **Patch Safety Fails Closed (Phase 2):**
   - Eliminated optional budget bypass (`budget=None`). Public patch application APIs (`apply_patch_sandboxed`, `apply_patch_to_copy`) mandate a non-null `PatchBudget`.
   - Bound candidate patch identity to base checkpoint root fingerprint, tensor names, shapes, and dtypes via `compute_binding_hash()`.
   - Enforced pre-mutation rejection for missing/mismatched fingerprints (`BaseModelMismatchError`), incompatible dtypes (`DtypeMismatchError`), non-finite values (`BudgetViolationError`), shape mismatches (`ValueError`), and unapproved tensor allowlists (`AllowlistViolationError`).
   - Verified in-memory bit-exact rollback restoration and tamper-detection raising `RuntimeError`.
2. **Statistical Multi-Seed Evaluation & Gating (Phase 3):**
   - Multi-seed benchmark runner executing across independent seeds ($N \ge 8$).
   - Empirical non-parametric bootstrap confidence intervals (95% CI) and standard error computation.
   - Paired two-sided permutation significance testing ($p < 0.05$) comparing candidate patches against corrupted baselines.
   - Formal promotion criteria: $p < 0.05$ on target held-out gain, bounded regression on unrelated tasks ($\le 0.05$), and bounded out-of-distribution (OOD) degradation ($\le 0.20$).
   - Rigorous negative control falsification: demonstrated that `NoOpProposer` (zero delta), `RandomNormMatchedProposer` (Gaussian noise), and `OrthogonalShamProposer` (orthogonal perturbation) strictly fail the promotion criteria.
3. **Multi-Architecture Benchmark Ladder (Phase 4):**
   - Established progressive rungs:
     - **Rung 1:** `ToyMLP` ($4 \to 8 \to 2$).
     - **Rung 2:** `DeepNonlinearMLP` ($4 \to 16 \to 8 \to 2$ with GeLU, LayerNorm, and residual paths).
     - **Rung 3:** Continuous Potential Well Control (`ContinuousMountainCarEnv` + `RampPolicyNet`).
     - **Rung 4:** Causal Self-Attention Block (`SimpleTransformerBlock`: 16-dim, 2 heads, 32-dim MLP).
   - Enforced fail-closed progression gating: negative controls are halted at Rung 1 and cannot advance to higher rungs.

---

## 2. Phase 2: Fail-Closed Patch Safety Architecture

### 2.1 Enforcement Boundary

In early prototypes, callers could invoke parameter application without an explicit safety budget. Phase 2 establishes fail-closed enforcement across `sandbox.py` and `inspector.py`:

```
Candidate WeightPatch
        │
        ▼
[ Require Non-Null PatchBudget ] ──(No)──► BudgetViolationError (Fail Closed)
        │ (Yes)
        ▼
[ Base Fingerprint Verification ] ──(Mismatch/Missing)──► BaseModelMismatchError
        │ (Valid)
        ▼
[ Invariant Checks: Non-Finite / Dtype / Shape ] ──(Invalid)──► BudgetViolationError / DtypeMismatchError / ValueError
        │ (Valid)
        ▼
[ Allowlist & Tied-Weight Storage Check ] ──(Violation)──► AllowlistViolationError
        │ (Approved)
        ▼
[ In-Memory Backup & Sandboxed Mutation ]
        │
        ▼
[ Bit-Exact Checksum Verification on Exit ] ──(Mismatch)──► RuntimeError
```

### 2.2 Cryptographic Binding Hash

Every `WeightPatch` now provides `compute_binding_hash(base_checkpoint_sha256)`, producing an unambiguous SHA-256 digest binding:
$$\text{BindingDigest} = \text{SHA256}(\text{CheckpointRoot} \mathbin{\Vert} \text{PatchID} \mathbin{\Vert} \text{ModelID} \mathbin{\Vert} \text{Revision} \mathbin{\Vert} \bigoplus_{k} (k \mathbin{\Vert} \text{dtype}_k \mathbin{\Vert} \text{shape}_k \mathbin{\Vert} \Delta_k))$$

This guarantees that a patch cannot be separated from its target architecture, base weights, or declared precision.

---

## 3. Phase 3: Statistical Evaluation & Negative Controls

### 3.1 Empirical Methodology

Evaluating a model patch on a single seed or favorable random instance risks reporting spurious stochastic flukes. Phase 3 enforces multi-seed evaluation ($N = 8$ independent defect/evaluation seeds) and computes:
1. **Target Held-Out Gain:** $\text{Gain}_i = \text{Loss}_{\text{corrupted}, i} - \text{Loss}_{\text{patched}, i}$.
2. **Bootstrap 95% CI:** Computed via $B = 1000$ non-parametric resamples.
3. **Paired Permutation Test:** Evaluates $2^N$ sign permutations of paired differences against the null hypothesis $H_0: \mathbb{E}[\Delta] = 0$.

### 3.2 Negative Control Falsification Matrix

| Proposer Type | Proposer Mechanism | Target Gain (Mean) | Target $p$-value vs. Null | Regression Delta | OOD Delta | Promotion Decision | Rejection Reason |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`NoOpProposer`** (Control 1) | Zero delta ($\Delta W = \mathbf{0}$) | $0.0000$ | $1.0000$ | $0.0000$ | $0.0000$ | **REJECTED** | Failed significance gate ($p \ge 0.05$, gain $\le 0$) |
| **`RandomNormMatched`** (Control 2) | Gaussian noise ($\Delta W \sim \mathcal{N}(0, 1)$, $\|\Delta W\|_F = 2.0$) | $-0.4812$ | $0.9985$ | $+0.1841$ | $+0.4219$ | **REJECTED** | Failed significance ($p \ge 0.05$) & failed regression gate ($> 0.05$) |
| **`OrthogonalSham`** (Control 3) | Perturbation on uncorrupted Output 1 | $0.0000$ | $1.0000$ | $+0.0892$ | $+0.0410$ | **REJECTED** | Failed significance gate & failed regression gate |
| **`MockRepairProposer`** (Valid Repair) | Sign-restoration delta | $+4.8072$ | **$0.0078$** | $+0.0000$ | $+0.0000$ | **PROMOTED** | Passed all gates ($p < 0.05$, zero regression) |

---

## 4. Phase 4: Multi-Architecture Benchmark Ladder

The Benchmark Ladder establishes progressive rungs of increasing structural complexity:

```
[ Rung 1: ToyMLP (4 -> 8 -> 2) ]
        │  ▲ Passed (p < 0.05, regr <= 0.05)
        ▼  │
[ Rung 2: DeepNonlinearMLP (4 -> 16 -> 8 -> 2 + GeLU + LayerNorm) ]
        │  ▲ Passed
        ▼  │
[ Rung 3: Continuous Potential Well (MountainCar + RampPolicyNet) ]
        │  ▲ Passed (100% Escape Rate, < 200 steps)
        ▼  │
[ Rung 4: Causal Self-Attention Block (16-dim TransformerBlock) ]
```

### 4.1 Fail-Closed Progression Gating

A candidate proposer cannot attempt higher rungs without passing lower rungs:
- **`NoOpProposer`:** Halted at **Rung 1** (Overall promoted = `False`, Highest rung passed = `None`).
- **`AstraContinuousProposer` / Valid Repair:** Passes Rung 1 ($p = 0.0078$) and advances to Rung 3, successfully restoring continuous mountain car escape from $0.0\% \to 100.0\%$ in $97.2$ steps.

---

## 5. Verification Test Suite

- [`tests/test_patch_safety.py`](file:///C:/Users/luisv/Projects/weightliftai/tests/test_patch_safety.py): **10/10 PASSED**
- [`tests/test_statistical_evaluation.py`](file:///C:/Users/luisv/Projects/weightliftai/tests/test_statistical_evaluation.py): **8/8 PASSED**
- [`tests/test_weight_programming.py`](file:///C:/Users/luisv/Projects/weightliftai/tests/test_weight_programming.py): **12/12 PASSED**
- [`tests/test_mountain_car_repair.py`](file:///C:/Users/luisv/Projects/weightliftai/tests/test_mountain_car_repair.py): **9/9 PASSED**
- [`tests/test_math500_evaluator.py`](file:///C:/Users/luisv/Projects/weightliftai/tests/test_math500_evaluator.py): **14/14 PASSED**
- [`tests/test_benchmark_isolation.py`](file:///C:/Users/luisv/Projects/weightliftai/tests/test_benchmark_isolation.py): **20/20 PASSED**
