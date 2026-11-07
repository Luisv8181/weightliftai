# Threat Model & Isolation Architecture for Blind Weight-Repair Benchmark

**Status:** Normative Specification for Benchmark Security, Evaluator Authority & Capability Boundaries  
**Version:** 2.0.0 (2026-11-07)  
**Tracking Issues:** [Issue #12](https://github.com/Luisv8181/weightliftai/issues/12), [Issue #14 (`WSP-001A`)](https://github.com/Luisv8181/weightliftai/issues/14)

---

## 1. Objective & Scope

The purpose of the Blind Weight-Repair Benchmark is to determine whether an optimization agent, LLM proposer, or repair algorithm can synthesize valid, generalizing neural weight interventions **purely from authorized diagnostic evidence**, without reconstructing or leaking ground-truth clean weights, defect masks, private random seeds, or sealed held-out evaluation splits.

Version 2.0.0 (`WSP-001A`) closes the remaining gaps identified in the November 6–7 council review:
1. **Evaluator-Authoritative Registration & Persistent Query Budget (`evaluator_authority.py`):** Query limits are no longer stored in a proposer-side `_queries_used` attribute. The evaluator generates an unpredictable `experiment_id`, binds it to an immutable `config_sha256`, enforces `allowed_submissions = 1` in a persistent SQLite ledger (`BEGIN IMMEDIATE` transactions) across multiple proposer instances, restarts, and concurrent calls, and consumes the attempt **before** patch validation (`CONSUME_BEFORE_VALIDATION`).
2. **Private Evaluator-Controlled Randomness (`defect_families.py`):** Benchmark instances are generated privately inside the evaluator using cryptographic randomness (`secrets.randbits(31)`), never from a public seed shared with the proposer. Clean weights cannot be deterministically reconstructed by rerunning training in the proposer process.
3. **Structured Pipe IPC (`ipc_protocol.py`):** Proposer-to-evaluator communication uses a versioned JSON protocol over `stdin`/`stdout` pipes (`--ipc-stdin`), eliminating shared temporary patch files.
4. **Pre-Allocation Resource Limits:** Request byte size, tensor count, rank, dimension lengths, parameter counts, and scalar finiteness (`NaN`/`Inf`/magnitude) are validated before allocating NumPy/PyTorch tensors.
5. **Standardized Error Categories & Cryptographic Receipts:** Proposer-visible errors use 5 standardized `ErrorCategory` codes without stack traces or internal paths. Valid evaluations return coarse 4-decimal aggregate metrics bound to an **Ed25519 digital signature** (`signed_receipt`) verifiable under the evaluator's public key.

---

## 2. Assets to Protect

| Asset | Description | Impact of Leakage | Protection Mechanism |
| :--- | :--- | :--- | :--- |
| **Clean Baseline Weights ($W^*$)** | Original uncorrupted parameter tensors. | **Catastrophic.** Trivial subtraction ($\Delta W = W^* - W_{\text{corrupt}}$) yields the oracle patch. | Generated & stored exclusively in the evaluator's private vault; never sent over IPC. |
| **Private Instance Seed ($s_{\text{priv}}$)** | Random seed governing data generation and clean model initialization. | **Catastrophic.** Allows deterministic regeneration of $W^*$ and all hidden splits. | Generated via `secrets.randbits(31)` inside the evaluator process; accessible only via `researcher_audit_replay` with `researcher_audit_token`. |
| **Defect Mask & Coordinates** | Exact corrupted indices and perturbation parameters. | **Severe.** Collapses fault-localization search space. | Stored only in the evaluator vault; proposer receives only diagnostic condition data. |
| **Sealed Held-Out / OOD / Regression Splits** | Independent evaluation tensors ($D_{\text{test}}, D_{\text{ood}}, D_{\text{reg}}$). | **Critical.** Enables direct overfitting to the test distribution. | Held exclusively inside the evaluator process; never serialized to proposer. |
| **Evaluator Ed25519 Private Key** | Signing key for evaluation receipts. | **Critical.** Enables forgery of passing benchmark results. | Generated & stored in evaluator SQLite vault; only `public_key_hex` is exposed. |

---

## 3. Trust Boundary & System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                       PROPOSER / AGENT DOMAIN                          │
│  (Untrusted: LLM proposer, gradient baseline, or external optimizer)   │
│                                                                        │
│  • Receives at Registration (scoped by DiagnosticCondition):           │
│    - Evaluator-generated `experiment_id` & `config_sha256`             │
│    - `evaluator_public_key` (Ed25519)                                  │
│    - Scoped diagnostic bundle (`behavioral_only`, `weights_only`,      │
│      `weights_and_failures`, `weights_activations_failures`,           │
│      or `full_diagnostic_toolkit`)                                     │
│                                                                        │
│  • Submits Once:                                                       │
│    - Candidate `WeightPatch` + `experiment_id` + `config_sha256`       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
          Structured Pipe IPC (`stdin`/`stdout`, Version 1.0, <= 2 MiB)
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  ISOLATED EVALUATOR AUTHORITY SERVICE                  │
│  (Trusted OS Subprocess + Persistent SQLite WAL Ledger)                │
│                                                                        │
│  1. Pre-Allocation Gate: Validates byte length, JSON schema, tensor    │
│     dimensions, parameter limits, and scalar finiteness (`NaN`/`Inf`). │
│  2. Atomic Ledger Gate (`BEGIN IMMEDIATE`):                            │
│     - Verifies `experiment_id`, expiration, and `config_sha256`.       │
│     - Enforces `CONSUME_BEFORE_VALIDATION` (`attempts_used += 1`).     │
│  3. Sandboxed Evaluation:                                              │
│     - Reconstructs private instance from vault (`private_seed`).       │
│     - Evaluates on sealed `held_out`, `ood`, and `regression` splits.  │
│  4. Controlled Feedback & Signing:                                     │
│     - Logs full-precision diagnostics to researcher-only audit table.  │
│     - Signs coarse aggregate metrics (4-decimal rounding) with Ed25519.│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Attempt Consumption, Error & Metric Release Policies

### 4.1 Attempt Policy (`CONSUME_BEFORE_VALIDATION`)
- Every `submit_candidate` call referencing a valid, active, unexpired `experiment_id` with a matching `config_sha256` **atomically increments `attempts_used` before patch validation or model execution**.
- Rationale: If failed or invalid patches did not consume an attempt, an untrusted proposer could probe internal validation boundaries or timing side channels indefinitely without exhausting its budget.

### 4.2 Standardized Proposer-Visible Error Categories
All evaluator failures return a structured JSON envelope with one of five `ErrorCategory` values and no Python tracebacks, file paths, or secret state:
- `AUTH_ERROR`: Unknown, expired, completed, or tampered `experiment_id`/`config_sha256`, unauthorized direct CLI invocation, or query budget exceeded.
- `PROTOCOL_ERROR`: Malformed UTF-8/JSON, unsupported `protocol_version`, unknown IPC action, missing required fields, or candidate SHA-256 mismatch.
- `RESOURCE_LIMIT_ERROR`: Payload `> 2 MiB`, tensor count `> 16`, rank `> 4`, dimension `> 4096`, total parameters `> 262,144`, numeric magnitude `> 1e6`, or subprocess/evaluation timeout.
- `VALIDATION_ERROR`: Base model fingerprint mismatch, ragged array, non-finite values (`NaN`/`Inf`), allowlist violation, or L2/relative norm budget violation.
- `INTERNAL_EVALUATOR_ERROR`: Generic message masking unexpected internal exceptions; detailed diagnostics are written only to the researcher audit log.

### 4.3 Metric Release & Side-Channel Reduction
- **Proposer-Visible Output:** Returns only `passed_gates` (bool), `recommendation` (`"promote"`/`"reject"`), `gate_thresholds`, 4-decimal rounded `gains`, 4-decimal rounded `aggregate_scores` (`held_out_mse`, `ood_mse`, `regression_mse`), and the Ed25519 `signed_receipt`.
- **Researcher-Only Audit Record:** Full floating-point precision, per-split baseline/patched breakdowns, wall-clock timing, `private_seed`, and validation internals are stored in `submission_audit_log` and require `researcher_audit_token` via `researcher_audit_replay`.

---

## 5. Legacy Entry Points vs. Sealed Boundary

| Module / Class | Isolation Status | Intended Use |
| :--- | :--- | :--- |
| `BlindProposerEnvironment` (`proposer_api.py`) + `isolated_evaluator.py` | **Sealed Evaluator-Authoritative Process Boundary** | All blind repair evaluations, LLM proposer runs, and diagnostic ablations (`WSP-001A`/`WSP-001B`). |
| `BlindRepairBenchmark` (`blind_benchmark.py`) | **`LEGACY_SAME_PROCESS_BASELINE_HARNESS` (`IS_SEALED_BLIND_BOUNDARY = False`)** | Same-process reference computation of `no_op`, `random_norm_matched`, `gradient_baseline`, and `oracle_upper_bound` for unit tests and CLI smoke checks (`weightliftai patch test-blind`). Never use for untrusted proposers. |

---

## 6. Technical Guarantees vs. Operational Assumptions & Residual Risks

1. **Guaranteed by Architecture:**
   - Clean weights, defect masks, private seeds, and hidden splits never exist in the proposer's Python process memory (`gc.get_objects()`).
   - Clean weights and hidden splits cannot be reconstructed from public seed hints (`seed=42`) and repository source code because the evaluator generates `private_seed` via `secrets.randbits(31)` inside the evaluator subprocess.
   - Creating new `BlindProposerEnvironment` instances, mutating `_queries_used` or `_max_queries`, or restarting the proposer process cannot bypass the per-experiment submission limit in the evaluator's SQLite ledger.
   - Forged or modified evaluation results fail `verify_signed_evaluation_result()` under the evaluator's Ed25519 public key.
2. **Operational & Deployment Limitations (Unsupported Attacker Classes):**
   - **Same-OS-User Filesystem Access:** In a local single-user development environment without OS user separation or containerization, a malicious program executing arbitrary filesystem reads as the same OS user could locate and read the SQLite ledger file on disk. Defending against arbitrary host filesystem reads requires running `isolated_evaluator.py` under a separate OS user/container or remote service boundary.
   - **OS Resource Controls on Windows vs. POSIX:** Pre-allocation byte/dimension checks, `torch.set_num_threads(1)`, and `subprocess. communicate(timeout=...)` are enforced cross-platform. Kernel `RLIMIT_AS` / `RLIMIT_CPU` limits apply only on POSIX hosts where `resource` is supported.
   - **Residual Timing Side Channels:** While error messages are standardized and attempts are capped at 1 per experiment (`CONSUME_BEFORE_VALIDATION`), wall-clock response time still differs slightly between fast pre-allocation rejection and full forward-pass evaluation.
