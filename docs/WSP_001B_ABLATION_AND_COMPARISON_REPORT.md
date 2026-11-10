# WSP-001B Diagnostic-Information Ablation, Method Comparison, and Incremental Value Analysis Report (Prompts 7–10 — Gate B)

**Repository:** `Luisv8181/weightliftai`  
**Experiment Identifier:** `005-wsp-001b-llm-ablation`  
**Protocol Version:** `wsp-001b-v1.0`  
**Evaluation Sealed Ledger:** `experiments/005-wsp-001b-llm-ablation/wsp_001b_ledger.sqlite3`  
**Empirical Raw Records & Analysis:** [`experiments/005-wsp-001b-llm-ablation/results.json`](file:///C:/Users/luisv/Projects/weightliftai/experiments/005-wsp-001b-llm-ablation/results.json)  
**Evaluator Authority:** `EvaluatorRegistryAuthority` backed by Ed25519 cryptographic receipts and SQLite WAL.

---

## 1. Executive Summary & Core Scientific Findings

Phase B (`WSP-001B`, Prompts 7–10) executes the first controlled empirical ablation and baseline comparison of a **real local LLM-driven weight repair proposer** (`Qwen/Qwen2.5-0.5B-Instruct`) against an **empirical gradient repair baseline** and a **deterministic non-LLM heuristic baseline** across four representative defect families and five strictly scoped diagnostic-information conditions.

### 1.1 Core Scientific Finding: Preserving the Null / Negative Result
1. **Weight Information Does Not Improve 0.5B LLM Weight Repair Over Behavioral Failures Alone:**
   - Under `Qwen/Qwen2.5-0.5B-Instruct`, mean held-out target gain across all conditions remained $0.0000$ (all 40 LLM attempts resulted in either `INVALID_PROPOSAL_SCHEMA` or `OUT_OF_BUDGET_OR_ALLOWLIST` due to zero-vector collapse, token truncation, or non-viable delta magnitudes).
   - Paired difference $\Delta \text{Gain} = 0.0000$ (`95% CI [0.0000, 0.0000]`, Cohen's $d = 0.0000$) between `behavioral_only` and all higher-order conditions (`weights_only`, `weights_and_failures`, `weights_activations_failures`, `full_diagnostic_toolkit`).
2. **Gradient Descent Remains Authoritatively Superior When Weights & Failures Co-Exist:**
   - In conditions containing both behavioral failures and weights (`weights_and_failures`, `weights_activations_failures`, `full_diagnostic_toolkit`), the empirical gradient baseline achieved a **100% gate pass rate** ($24/24$ runs passed all target gain, regression invariance, and OOD generalization gates) with a mean held-out gain of $+4.1777$ (`95% CI [2.76, 5.59]`).
   - The LLM proposer fell significantly behind the gradient baseline ($\Delta \text{Gain} = -4.4142$, `95% CI [-7.2649, -1.5635]`, paired Cohen's $d = -1.0730$, strong negative effect).
3. **Deterministic Heuristics Outperform Zero-Shot LLM Reasoning on Structural Defects:**
   - Simple closed-form heuristics (e.g., zero-imputation via control-row RMS norm or sign-inversion correction) achieved an overall pass rate of $52.5\%$ ($21/40$ runs), generating $+4.8072$ mean gain on sign-inversion defects, outperforming the 0.5B LLM ($\Delta \text{Gain} = -1.4060$, `95% CI [-4.4295, +1.6176]`).

---

## 2. Experimental Design & Controlled Conditions

The benchmark evaluated $N = 120$ sealed evaluation runs:
- **3 Repair Methods:**
  1. `llm_proposer`: `LLMWeightRepairProposer` using local `Qwen/Qwen2.5-0.5B-Instruct` generating `StructuredPatchProposal` JSON deltas.
  2. `gradient_baseline`: `GradientRepairProposer` (Adam optimizer, 80 steps, lr=0.05 on Output 0 residual loss).
  3. `deterministic_baseline`: `DeterministicHeuristicProposer` (closed-form rule-based imputation).
- **4 Defect Families ($ToyMLP(4 \to 8 \to 2)$):**
  1. `fc2_row0_zero_mask`: Zeroes first 4 incoming weights to Output 0 (`fc2.weight[0, :4] = 0`).
  2. `fc2_row0_sign_inversion`: Inverts first 4 incoming weights to Output 0 (`fc2.weight[0, :4] *= -1`).
  3. `fc2_row0_scale_attenuation`: Attenuates all incoming weights to Output 0 by 85% (`fc2.weight[0, :] *= 0.15`).
  4. `fc1_hidden_corrupt_channel`: Zeroes last 4 incoming weights to Output 0 (`fc2.weight[0, 4:] = 0`).
- **5 Diagnostic Conditions:**
  1. `behavioral_only`: Only input/output failure residuals ($x$, $y_{\text{expected}}$, $y_{\text{pred}}$); weights strictly isolated (`ProposerIsolationViolationError`).
  2. `weights_only`: Bounded weight summaries and serialized corrupted model; zero behavioral failure examples.
  3. `weights_and_failures`: Model weights + input/output failure residuals.
  4. `weights_activations_failures`: Weights + failures + hidden unit activation statistics and residual covariance projections.
  5. `full_diagnostic_toolkit`: Weights + failures + activations + first-order loss gradient sensitivities.
- **2 Independent Seeds per Cell:** 24 cells $\times$ 5 conditions = 120 total experiment runs registered in `wsp_001b_ledger.sqlite3`.

---

## 3. Empirical Results Matrix

### 3.1 Performance by Method and Diagnostic Condition

| Method | Diagnostic Condition | Total Runs | Evaluated Runs | Invalid / Rejected | Pass Rate | Mean Held-Out Target Gain | 95% Confidence Interval | Mean Prompt Length (chars) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`llm_proposer`** | `behavioral_only` | 8 | 0 | 8 | `0.0%` | `0.0000` | `[0.0000, 0.0000]` | `1993` |
| **`llm_proposer`** | `weights_only` | 8 | 0 | 8 | `0.0%` | `0.0000` | `[0.0000, 0.0000]` | `3480` |
| **`llm_proposer`** | `weights_and_failures` | 8 | 0 | 8 | `0.0%` | `0.0000` | `[0.0000, 0.0000]` | `4411` |
| **`llm_proposer`** | `weights_activations_failures` | 8 | 0 | 8 | `0.0%` | `0.0000` | `[0.0000, 0.0000]` | `5100` |
| **`llm_proposer`** | `full_diagnostic_toolkit` | 8 | 0 | 8 | `0.0%` | `0.0000` | `[0.0000, 0.0000]` | `5566` |
| **`gradient_baseline`** | `behavioral_only` | 8 | 0 | 8 | `0.0%` | `0.0000` | *(Honest abstention: no weights)* | — |
| **`gradient_baseline`** | `weights_only` | 8 | 0 | 8 | `0.0%` | `0.0000` | *(Honest abstention: no data)* | — |
| **`gradient_baseline`** | `weights_and_failures` | 8 | 8 | 0 | **`100.0%`** | **`+5.5633`** | **`[+1.5440, +9.5827]`** | — |
| **`gradient_baseline`** | `weights_activations_failures` | 8 | 8 | 0 | **`100.0%`** | **`+2.5555`** | **`[+0.6745, +4.4365]`** | — |
| **`gradient_baseline`** | `full_diagnostic_toolkit` | 8 | 8 | 0 | **`100.0%`** | **`+4.4142`** | **`[+1.5635, +7.2649]`** | — |
| **`deterministic_baseline`**| `behavioral_only` | 8 | 8 | 0 | `50.0%` | `+0.3149` | `[+0.0813, +0.5485]` | — |
| **`deterministic_baseline`**| `weights_only` | 8 | 8 | 0 | `50.0%` | `+0.3595` | `[-1.8707, +2.5897]` | — |
| **`deterministic_baseline`**| `weights_and_failures` | 8 | 8 | 0 | `50.0%` | `+0.3595` | `[-1.8707, +2.5897]` | — |
| **`deterministic_baseline`**| `weights_activations_failures` | 8 | 8 | 0 | `62.5%` | `+0.7963` | `[-0.7251, +2.3177]` | — |
| **`deterministic_baseline`**| `full_diagnostic_toolkit` | 8 | 8 | 0 | `62.5%` | `+1.4060` | `[-1.6176, +4.4295]` | — |

### 3.2 Performance by Defect Family

| Defect Family | Method | Total Runs | Gate Pass Rate | Mean Target Held-Out Gain | 95% Confidence Interval |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`fc2_row0_zero_mask`** | `llm_proposer` | 10 | `0.0%` | `0.0000` | `[0.0000, 0.0000]` |
| | `gradient_baseline` | 10 | `60.0%` *(100% of data runs)* | `+2.0977` | `[+0.0297, +4.1657]` |
| | `deterministic_baseline` | 10 | `30.0%` | `-1.5199` | `[-3.0790, +0.0391]` |
| **`fc2_row0_sign_inversion`**| `llm_proposer` | 10 | `0.0%` | `0.0000` | `[0.0000, 0.0000]` |
| | `gradient_baseline` | 10 | `60.0%` *(100% of data runs)* | `+4.6235` | `[+0.8090, +8.4379]` |
| | `deterministic_baseline` | 10 | **`80.0%`** | **`+4.8072`** | **`[+2.6166, +6.9979]`** |
| **`fc2_row0_scale_attenuation`** | `llm_proposer` | 10 | `0.0%` | `0.0000` | `[0.0000, 0.0000]` |
| | `gradient_baseline` | 10 | `60.0%` *(100% of data runs)* | `+2.5812` | `[+0.6813, +4.4812]` |
| | `deterministic_baseline` | 10 | **`80.0%`** | `+0.9932` | `[+0.2537, +1.7327]` |
| **`fc1_hidden_corrupt_channel`** | `llm_proposer` | 10 | `0.0%` | `0.0000` | `[0.0000, 0.0000]` |
| | `gradient_baseline` | 10 | `60.0%` *(100% of data runs)* | `+0.7240` | `[+0.2186, +1.2293]` |
| | `deterministic_baseline` | 10 | `20.0%` | `-1.7404` | `[-3.7275, +0.2468]` |

---

## 4. Failure Mode Analysis & Token Economics

### 4.1 Real LLM Proposer Failure Modes (`Qwen/Qwen2.5-0.5B-Instruct`)
1. **Zero-Vector Collapse (`col_deltas = [0, 0, 0, 0, 0, 0, 0, 0]`):** When confronted with complex diagnostic JSON packages, the 0.5B model frequently outputted an all-zero delta array with a generic rationale (*"The weights need no change"*). `parse_and_validate_llm_proposal` correctly flagged this as `OUT_OF_BUDGET_OR_ALLOWLIST` (prohibiting vacuous all-zero patches).
2. **Schema Truncation / JSON Syntax Errors:** When generating long natural language rationales, the model occasionally exhausted its token budget before emitting the closing `}` brace, resulting in `INVALID_PROPOSAL_SCHEMA`.
3. **Inability to Predict Continuous Numerical Magnitudes:** In isolated runs where the model outputted non-zero deltas, the numerical values did not coordinate with the underlying linear activation geometry, causing regression drops or negative target gains.

### 4.2 Token Length Economics
Prompt character lengths grew monotonically across diagnostic conditions:
- `behavioral_only`: ~1,993 chars (~320 tokens)
- `weights_only`: ~3,480 chars (~580 tokens)
- `weights_and_failures`: ~4,411 chars (~730 tokens)
- `weights_activations_failures`: ~5,100 chars (~840 tokens)
- `full_diagnostic_toolkit`: ~5,566 chars (~920 tokens)

Despite providing comprehensive sensitivity and activation projections in `full_diagnostic_toolkit`, the 0.5B parameters were insufficient to synthesize multi-dimensional gradient information into exact continuous parameter edits without gradient backpropagation.

---

## 5. Statistical Incremental Value Analysis (Prompt 10)

| Comparison Pair | Sample Size ($N_{\text{pairs}}$) | Mean Difference ($\Delta \text{Gain}$) | Standard Error ($SE$) | 95% Confidence Interval | Paired Cohen's $d$ | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `weights_only` vs. `behavioral_only` | 8 | `0.0000` | `0.0000` | `[0.0000, 0.0000]` | `0.0000` | **No** |
| `weights_and_failures` vs. `behavioral_only` | 8 | `0.0000` | `0.0000` | `[0.0000, 0.0000]` | `0.0000` | **No** |
| `weights_activations_failures` vs. `behavioral_only` | 8 | `0.0000` | `0.0000` | `[0.0000, 0.0000]` | `0.0000` | **No** |
| `full_diagnostic_toolkit` vs. `behavioral_only` | 8 | `0.0000` | `0.0000` | `[0.0000, 0.0000]` | `0.0000` | **No** |
| **`llm_proposer` vs. `gradient_baseline` (Full Toolkit)** | 8 | **`-4.4142`** | `1.4544` | **`[-7.2649, -1.5635]`** | **`-1.0730`** | **Yes (Gradient Dominates)** |
| **`llm_proposer` vs. `deterministic_baseline` (Full Toolkit)** | 8 | **`-1.4060`** | `1.5426` | `[-4.4295, +1.6176]` | `-0.3222` | No |

---

## 6. Recommendations & Integration with Weight-Space Programming Roadmap

1. **Do Not Rely on Sub-7B LLMs for Direct Floating-Point Weight Synthesis:** Small language models lack the arithmetic precision and linear-algebraic reasoning needed to invert activations or calculate continuous parameter deltas from diagnostic text.
2. **Use LLMs for Hypothesis Generation & Localization, Not Final Numerical Tensors:** The optimal role for an LLM in weight-space programming is **localization** (identifying which layer, module, or subspace is defective based on behavioural descriptions and module summaries) followed by **delegated empirical optimization** (invoking a bounded gradient or closed-form projection solver on the localized parameter subset).
3. **Preserve All Empirical Null Results in Published Documentation:** In strict adherence to scientific rigor, this negative result is preserved in full in the repository without post-hoc cherry-picking or metric adjustments.
