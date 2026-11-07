# Threat Model & Isolation Architecture for Blind Weight-Repair Benchmark

**Status:** Normative Specification for Benchmark Security & Capability Boundaries  
**Version:** 1.0.0 (2026-11-06)  
**Tracking Issue:** [Issue #12: Harden blind weight-repair benchmark isolation, patch safety, and statistical evaluation](https://github.com/Luisv8181/weightliftai/issues/12)

---

## 1. Objective & Scope

The purpose of the Blind Weight-Repair Benchmark is to determine whether an optimization agent or algorithm can synthesize valid, generalizing neural weight interventions **purely from diagnostic failure evidence**, without cheating or leaking ground-truth clean weights, defect locations, or sealed held-out evaluation splits.

A benchmark that claims to be "blind" while retaining secret test splits in the same Python process/object graph is vulnerable to trivial introspection (`self.__dict__`, `gc.get_objects()`, reflection). This document formalizes the assets, threat boundaries, attack vectors, and process isolation architecture enforced in **WeightLiftAI**.

---

## 2. Assets to Protect

The benchmark environment guards four distinct classes of confidential assets from the repair proposer:

| Asset | Description | Impact of Leakage |
| :--- | :--- | :--- |
| **Clean Baseline Weights** | The original uncorrupted parameter tensors ($W^*$). | **Catastrophic.** Trivial subtraction ($\Delta W = W^* - W_{\text{corrupt}}$) recovers oracle upper bounds with zero learning. |
| **Defect Mask & Slices** | Coordinates and mechanisms of artificial defects (e.g. `fc2.weight[0, :4] = 0.0`). | **Severe.** Proposer restricts search to the exact corrupted sub-tensor without localized diagnostic reasoning. |
| **Sealed Held-Out Splits** | Independent test inputs, labels, and target evaluation tasks ($D_{\text{test}}$). | **Critical.** In-sample overfitting or direct empirical gradient fitting on test samples. |
| **OOD & Regression Splits** | Distribution-shifted and untouched capability verification benchmarks ($D_{\text{ood}}, D_{\text{reg}}$). | **Severe.** Metric gaming; tailoring interventions to artificially avoid regression penalties. |

---

## 3. Trust Boundary & System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 PROPOSER / AGENT DOMAIN                     │
│  (Untrusted: Executes exploratory search & optimization)    │
│                                                             │
│  • Reads:                                                   │
│    - Read-only corrupted model / checkpoint                 │
│    - High-level architecture metadata (layer names, shapes) │
│    - Diagnostic failure examples & error metrics ($D_{\text{diag}}$)│
│    - Pre-registered PatchBudget                             │
│                                                             │
│  • Submits:                                                 │
│    - Frozen candidate `WeightPatch` (one-way protocol)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
               Process Boundary (IPC / JSON Stdio)
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                ISOLATED EVALUATOR SERVICE                   │
│  (Trusted: Executes in a separate OS process address space) │
│                                                             │
│  • Privately holds:                                         │
│    - Ground-truth clean weights ($W^*$)                     │
│    - Defect creation mask and perturbation parameters       │
│    - Sealed test, OOD, and regression splits                │
│                                                             │
│  • Evaluates:                                               │
│    - Applies candidate patch to sandboxed replica           │
│    - Computes macro aggregate gains ($\Delta_{\text{target}}, \Delta_{\text{ood}}$) │
│                                                             │
│  • Returns:                                                 │
│    - Coarse aggregate scalar metrics ONLY                   │
│    - ZERO per-example predictions, losses, or gradients     │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Threat Vectors & Mitigations

### 4.1 In-Memory Object Traversal & Python Introspection
- **Threat:** Proposer probes `benchmark.__dict__`, traverses `gc.get_objects()`, inspects closure free variables (`func.__closure__`), or imports internal evaluator modules to read hidden tensors directly from memory.
- **Mitigation:** The Proposer API (`BlindProposerEnvironment`) **never instantiates or imports** the evaluator in its own Python address space. The evaluator executes strictly inside an isolated child subprocess. The proposer process contains zero memory references to clean weights or hidden test tensors.

### 4.2 Adaptive Hill-Climbing on Hidden Test Splits
- **Threat:** Proposer iteratively queries the hidden test evaluation endpoint with slight perturbations, effectively taking numerical derivatives through the evaluation API to reconstruct the test set or overfit.
- **Mitigation:** 
  1. **Strict Query Budget:** Proposers are granted exactly **one** evaluation submission per candidate experiment run.
  2. **Freeze-Before-Evaluation:** The candidate patch is cryptographically hashed, serialized, and sealed *before* invoking the evaluation subprocess. Post-hoc adaptation is prohibited.

### 4.3 Output Leakage via Granular Residuals
- **Threat:** If the evaluator returns individual per-example predictions, logits, residuals, or loss breakdowns, the proposer can solve a system of linear equations to infer the hidden test labels.
- **Mitigation:** The evaluator returns **only coarse aggregate metrics** (`target_gain`, `distribution_shift_gain`, `regression_drop`, `passed_gates`). Per-sample predictions, logits, and intermediate states are never returned across the process boundary.

### 4.4 Filesystem & Path Discovery Leakage
- **Threat:** The evaluator creates temporary files at predictable paths (e.g. `/tmp/sealed_test.pt`) that the proposer process could read from disk.
- **Mitigation:** Any temporary files used for inter-process serialization use cryptographically random UUIDs within isolated per-run directories with restricted read permissions, and are deleted immediately upon completion.

---

## 5. Proposer API Contract

The untrusted proposer interacts strictly with the following minimal interface:

```python
class BlindProposerEnvironment:
    """Narrow interface presented to the repair proposer."""
    
    @property
    def corrupted_model(self) -> torch.nn.Module:
        """Returns a cloned copy of the corrupted model."""
        ...
        
    @property
    def architecture_metadata(self) -> dict[str, Any]:
        """Returns module names, shapes, and parameter counts."""
        ...
        
    @property
    def diagnostic_failures(self) -> dict[str, Any]:
        """Returns observed diagnostic inputs and expected target labels."""
        ...
        
    @property
    def patch_budget(self) -> PatchBudget:
        """Returns allowable tensor names and norm budgets."""
        ...
        
    def submit_candidate_patch(self, patch: WeightPatch) -> dict[str, Any]:
        """Freezes the candidate patch, calls the isolated evaluator subprocess,
        and returns aggregate evaluation metrics."""
        ...
```

---

## 6. Residual Risks & Honest Disclosures

1. **Host-Level Isolation:** Process isolation (running a child Python process) prevents memory sharing, object graph traversal, and module reflection within Python. It **does not** constitute a hardened hypervisor or kernel sandbox against malicious C extensions, ptrace memory debugging, or root-level host attacks.
2. **Deterministic Pilot Scope:** This threat model governs scientific agent behavior to prevent unintentional evaluation leakage; it assumes the code being evaluated does not contain deliberately adversarial binary exploits.
