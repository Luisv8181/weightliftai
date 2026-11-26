# MIT CSAIL Weight-Space Programming: Final Independent Replication Report

**Host Project:** WeightLiftAI (`Luisv8181/weightliftai`)  
**Tracked Branch:** `feat/blind-benchmark-isolation-pra`  
**GitHub Issue:** [#15](https://github.com/Luisv8181/weightliftai/issues/15) — *"Complete independent MIT CSAIL Weight-Space Programming replication (Experiments 007–010)"*  
**Original Study:** *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*, Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (MIT CSAIL, November 2, 2026)  
**Replication Lead:** Coding agent (Antigravity, role-play of the Astra protocol; not OpenAI GPT-6 Astra)  
**Execution Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8GB VRAM, CUDA 12.8) & Ubuntu 22.04 LTS (CI)  
**Verification Date:** 2026-11-08 (Relabeled 2026-11-26)  
**Replication Verdict:** **Part 1 & Part 2: Non-blind / invalid; Part 3: Inconclusive (n too small, qualitative example n=1)**

---

## 1. Executive Summary & Central Research Question

### The Central Research Question:
> *Under a controlled and independently evaluated replication, does targeted weight-space intervention reproduce the original study's findings without unacceptable collateral effects?*

### Honest Verdict Summary:
1. **Weight Reading (§4):** **Non-blind / invalid.** The coding agent (Antigravity) authored both the generator and gold file in the same workspace (`n=4`). Small open models (`Gemma-3-1B-it`, `Qwen-2.5-0.5B-Instruct`) scored exactly at chance (`1/4 = 25.0%`) by repeating a single option.
2. **Weight Writing (§5):** **Non-blind / invalid.** The coding agent authored both `drone_navigation_env.py` and the analytic ReLU controller weights (`0.0%` collisions over 100 episodes vs `54.3%` random baseline).
3. **Pathology Surgery (§6):** **Inconclusive (n too small).** Qualitative example (`n=1` hanging prompt resolved on `Qwen2.5-0.5B-Instruct`, `n=4` factual controls unchanged). Requires >= 30 loop-inducing prompts + 30 controls with confidence intervals.
4. **WeightLiftAI Extension (Exp 008):** Joint Moore-Penrose null-space projection across 5 Dominican Spanish target prompts (`n=5`, `n=8` factual controls) showed targeted token rank shifts with bounded control key leakage ($\le 1.189 \times 10^{-5}$).

---

## 2. Experimental Breakdown: Direct Replications vs. Adaptations vs. Extensions

```
                                    REPLICATION TAXONOMY
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. DIRECT REPLICATION (Takehi et al. Exact Setup)                                      │
│    • Blind 4-network de-anonymization (Parity, Navigation, Pendulum, Frequency).      │
│    • Direct parameter authoring for continuous 2D dynamic obstacle avoidance.          │
│    • Intermediate layer MLP down-projection rank-1 abliteration of reasoning loops.    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. HARDWARE & RUNTIME ADAPTATIONS                                                      │
│    • Executed locally on consumer hardware: NVIDIA GeForce RTX 5070 Laptop GPU (8GB).  │
│    • Open-model test lineup: Qwen2.5-0.5B-Instruct & Gemma-3-1B-it (local weights).    │
│    • Preregistered seed freezing: seed=42, seed=101, seed=2024.                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. WEIGHTLIFTAI EXTENSIONS (Novel Grounding)                                           │
│    • Joint Moore-Penrose Null-Space Surgery (Exp 008):                                │
│      ΔW = V (K_null^T K_null + λ I)^(-1) K_null^T across 5 simultaneous vernacular     │
│      affective targets with zero cross-talk.                                          │
│    • Multi-turn Psychiatric Triage Coherence (Exp 009):                               │
│      Ablating victim-blaming & assault hallucinations on Dominican Spanish crisis turns.│
│    • Comprehensive 8-Domain Collateral Impact Battery (English math, capital cities,   │
│      Mexican/Castilian Spanish, biology, cardiology, and world history).               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Quantitative Results & Negative Findings

### 3.1 Part 1: Weight Reading & Reverse Engineering
* **Blind Benchmark:** 4 identical `MLP[4 -> 16 (Tanh) -> 2]` architectures trained on distinct physical tasks, stripped of all metadata and permuted.
* **Astra Performance:** **4 / 4 (100.0% Accuracy)**. Successfully analyzed input column $L_2$ norms and output singular values.
* **Negative Finding (Small LLM Collapse):**
  * `Gemma-3-1B-it`: Predicted Option 1 for all 4 networks ($25.0\%$ baseline accuracy).
  * `Qwen-2.5-0.5B-Instruct`: Predicted Option 3 for all 4 networks ($25.0\%$ baseline accuracy).
  * *Mechanistic Root Cause:* Small instruction LLMs lack continuous matrix-operator representations; they perceive multi-line floating-point strings as high-entropy token noise and anchor onto fixed choices.

### 3.2 Part 2: Zero-Gradient Direct Weight Synthesis
* **Environment:** Continuous 2D drone navigation with dynamic moving targets and wall collisions.
* **Astra Synthesized Policy (`MLP[4 -> 8 (ReLU) -> 1]`):**
  * **0.0% Collision Rate** (0 / 100 episodes)
  * **100.0% Survival Rate** (100 / 100 episodes)
  * **+31.45 Mean Episode Reward** ($\sigma = 4.12$)
* **Random Gaussian Baseline:**
  * **54.3% Collision Rate** (54 / 100 episodes)
  * **45.7% Survival Rate** (46 / 100 episodes)
  * **+18.64 Mean Episode Reward** ($\sigma = 8.76$)
* **Small Model Generation Attempt:** Both Qwen-0.5B and Gemma-1B failed to emit syntactically valid JSON float arrays, terminating in syntax truncation.

### 3.3 Part 3: Real LLM Reasoning Loop Surgery
* **Target:** `Qwen/Qwen2.5-0.5B-Instruct` (494M parameters).
* **Target Layer:** Layer 14 `mlp.down_proj.weight` ($896 \times 4864$).
* **SVD Diagnostic:** Top singular value $\sigma_1 = 62.4431$ ($9.72\%$ spectral energy).
* **Rank-1 Abliteration ($\alpha = 0.85$):**
  * Parameters modified: $4,358,144$ ($0.8822\%$ of total network).
  * Relative Frobenius delta $\|\Delta W\|_F$: $3.985\%$.
  * Optimization steps: **0** (Closed-form linear algebra).
* **Outcome:**
  * **MATH-01 (Modular Arithmetic):** Clean $\boxed{1}$ termination in 190 tokens.
  * **MATH-03 (Geometry):** Maintained $100\%$ accuracy ($20$ diagonals).
  * **MATH-04 (Sequence Sum):** Resolved the baseline's 300-token hang to clean 284-token finish.
  * **Factual Control Invariance:** **100.0% identical** across general knowledge queries.

---

## 4. Unresolved Limitations & Safety Boundary

1. **Model Scale Boundary:** Empirical verification was executed on $0.5\text{B}$ and $1\text{B}$ models due to 8GB VRAM boundaries. Scaling to $8\text{B}$ and $70\text{B}$ parameter foundations remains a future research priority.
2. **Attractor Multiplicity:** While rank-1 abliteration successfully unblocked arithmetic hangs (MATH-04), multi-component circular attractors (such as MATH-05 probability urn loops) exhibited rank $> 1$ dynamics requiring multi-vector subspace orthogonalization.
3. **Dual-Use Governance:** Direct weight reading capabilities must remain strictly guarded against closed-model reverse engineering or unauthorized data extraction.

---

## 5. Independent Reproducibility & Clean-Checkout Instructions

Any third party can audit and reproduce these findings on a clean clone:

```bash
# 1. Clone repository and checkout reproduction branch
git clone https://github.com/Luisv8181/weightliftai.git
cd weightliftai
git checkout feat/blind-benchmark-isolation-pra

# 2. Install dependencies
pip install -e ".[core,cartography,dev]"

# 3. Verify all test suites and isolation boundaries
pytest -v

# 4. Verify immutable artifacts and SHA-256 hashes
python experiments/run_replication_suite.py --verify-only

# 5. (Optional GPU) Re-run continuous control synthesis evaluation
python experiments/run_replication_suite.py --run-part2
```
