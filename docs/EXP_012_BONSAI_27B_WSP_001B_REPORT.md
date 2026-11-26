# Experiment 012: Bonsai-27B (1-Bit Local Ollama Model) on Sealed WSP-001B Blind Repair

**Repository:** `Luisv8181/weightliftai`  
**Date:** 2026-11-26  
**Model Under Test:** `bonsai-27b-q1:latest` (27B native 1-bit GGUF running locally in Ollama at `127.0.0.1:11434`, `think=False`, `temperature=0.0`, `repeat_penalty=1.15`)  
**Evaluator Authority:** `EvaluatorRegistryAuthority` (`WSP-001A` sealed SQLite WAL ledger, private CSPRNG seeds, `CONSUME_BEFORE_VALIDATION` attempt budget, Ed25519 signed receipts)  
**Raw Results & Ledger:** `experiments/012-bonsai-27b-wsp-001b/results.json` and `experiments/012-bonsai-27b-wsp-001b/bonsai_27b_ledger.sqlite3`  
**Impact on `bonsai-chat`:** Zero files touched in `bonsai-chat`; read-only HTTP calls to Ollama (`127.0.0.1:11434`).

---

## 1. Research Question

Following Fan et al. (*Language Models Can Rewrite Their Own Weights*, `arXiv:2611.21885`, Nov 2026), which reported that gradient-free self-rewriting emerges around the 27B parameter scale, and Experiment 005 (`WSP-001B`), where `Qwen/Qwen2.5-0.5B-Instruct` scored `0/40` (`0.0%`), we tested whether a **local 27B 1-bit model (`bonsai-27b-q1:latest`)** crosses the sealed `WSP-001B` repair threshold under two interfaces:

1. **`bonsai_27b_direct_json`:** Direct floating-point JSON `StructuredPatchProposal` (apples-to-apples comparison against `Qwen2.5-0.5B` in Experiment 005).
2. **`bonsai_27b_projection_solver`:** `SandboxedProjectionLLMProposer`, where the 27B model diagnoses the defect from the authorized diagnostic package and selects a bounded, gradient-free algebraic operation (`least_squares_projection`, `sign_flip_indices`, `scale_row`, or `explicit_deltas`) with zero `eval`/`exec` and zero access to private evaluator secrets.

Both were compared on the exact same sealed evaluator against `deterministic_baseline` and `gradient_baseline` across 4 defect families (`fc2_row0_zero_mask`, `fc2_row0_sign_inversion`, `fc2_row0_scale_attenuation`, `fc1_hidden_corrupt_channel`), 3 diagnostic conditions (`behavioral_only`, `weights_only`, `full_diagnostic_toolkit`), and 2 independent runs per cell (`96` sealed runs total, completed in `410.29s`).

---

## 2. Empirical Results (All 96 Sealed Runs Preserved)

### 2.1 Overall Summary by Method

| Method | Total Runs | Evaluated Runs | Invalid / Rejected / Abstained | Sealed Gate Passes | Pass Rate | Mean Held-Out Gain (All Runs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`bonsai_27b_direct_json`** | 24 | 15 | 9 | **2 / 24** | **8.3%** | `-8.4899` |
| **`bonsai_27b_projection_solver`** | 24 | 11 | 13 | **7 / 24** | **29.2%** *(63.6% of evaluated)* | `-6.5185` |
| **`deterministic_baseline`** | 24 | 24 | 0 | **16 / 24** | **66.7%** | `+1.6075` |
| **`gradient_baseline`** | 24 | 8 | 16 *(honest abstention)* | **8 / 24** | **33.3%** *(100% of full-toolkit runs)* | `+0.9329` |

### 2.2 Breakdown by Diagnostic Condition (8 Runs per Cell)

| Method | `behavioral_only` (Pass / Eval / Mean Gain) | `weights_only` (Pass / Eval / Mean Gain) | `full_diagnostic_toolkit` (Pass / Eval / Mean Gain) |
| :--- | :---: | :---: | :---: |
| **`bonsai_27b_direct_json`** | `1/8` (`5` eval, `-3.0516`) | `1/8` (`8` eval, `-2.1193`) | `0/8` (`2` eval, `-20.2988`) |
| **`bonsai_27b_projection_solver`** | `0/8` (`0` eval, `0.0000`) | **`6/8` (`75.0%`)** (`8` eval, `-0.1615`) | `1/8` (`3` eval, `-19.3940`) |
| **`deterministic_baseline`** | `4/8` (`8` eval, `+0.8629`) | `5/8` (`62.5%`) (`8` eval, `+1.9798`) | `7/8` (`8` eval, `+1.9797`) |
| **`gradient_baseline`** | `0/8` *(no weights)* | `0/8` *(no data)* | **`8/8` (`100.0%`)** (`8` eval, **`+2.7988`**) |

---

## 3. Key Findings & Failure Mode Diagnosis

1. **Overall Result: Simple Deterministic and Gradient Baselines Clearly Beat Bonsai-27B:**
   - Overall, `bonsai_27b_projection_solver` passed **`7/24` (`29.2%`)** with a **negative mean held-out gain (`-6.5185`)** (meaning its patches hurt performance on average across all runs), compared to **`16/24` (`66.7%`, `+1.6075` mean gain)** for `deterministic_baseline` and **`8/24` (`33.3%`, `+0.9329` mean gain; `8/8` on full toolkit)** for `gradient_baseline`.
   - `bonsai_27b_direct_json` passed only **`2/24` (`8.3%`, `-8.4899` mean gain)**.
   - In the `weights_only` cell (`n=8`), `bonsai_27b_projection_solver` scored `6/8` vs `5/8` for `deterministic_baseline` (not a statistically meaningful difference at `n=8`, and `deterministic_baseline` still had a much higher mean gain `+1.9798` vs `-0.1615`).
   - Compared to `Qwen2.5-0.5B` (`0/40` in Exp 005), 27B 1-bit moved above zero (`7/24`), but still trails simple baselines overall.
2. **Thinking-Token Budget Trap (`think=False` Requirement):**
   - An initial run without `think=False` produced `0/48` non-empty responses because `bonsai-27b-q1:latest` spent the entire `num_predict` budget inside Ollama's hidden thinking channel. Setting `think=False` reduced latency to ~1.15s–8s per call and unlocked structured JSON output.
3. **Two Concrete Failure Modes Preserved in the Raw Logs:**
   - **Direct Float Overshoot:** Even at 27B, directly emitting 8 floating-point numbers in JSON (`bonsai_27b_direct_json`) frequently overshot target magnitudes, causing large negative held-out gains (`-8.49` mean across all runs).
   - **Nullable Schema Friction (`ridge_lambda: null`):** In `full_diagnostic_toolkit`, `bonsai_27b_projection_solver` emitted `"ridge_lambda": null` on several runs, which Pydantic rejected as `INVALID_PROPOSAL_SCHEMA` before the closed-form `least_squares_projection` could execute.
4. **Gradient Descent Remains Dominant When Weights and Failures Co-Exist:**
   - On `full_diagnostic_toolkit`, `gradient_baseline` achieved `8/8` (`100.0%`, mean gain `+2.7988`).

---

## 4. Epistemic Classification

- **Tier 1 (Verified Facts):** All 96 sealed runs executed against live `bonsai-27b-q1:latest` in Ollama, recorded in `bonsai_27b_ledger.sqlite3` with Ed25519 signed receipts, and preserved in `results.json`.
- **Tier 2 (Test-Supported Findings):** On `ToyMLP`, moving from 0.5B (`0/40`) to 27B 1-bit (`7/24` with algebraic primitives; `6/8` under `weights_only`) produces real, sealed gate passes, while direct float guessing remains unreliable (`-8.49` mean gain).
- **Tier 4 (Limitations):** Tested on `ToyMLP(4 -> 8 -> 2)` with `think=False` and strict non-nullable `ridge_lambda`. Allowing nullable optional fields in `StructuredProjectionPlan` and testing `bonsai-27b-q1` on `experiments/helper_bench/` are clear next experiments.
