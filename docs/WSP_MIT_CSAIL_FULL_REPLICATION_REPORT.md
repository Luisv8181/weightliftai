# Sparks of Weight-Space Programming: Pilot Replication Report (Relabeled)

**Original Paper:** *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*  
**Authors:** Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (MIT CSAIL, Nov 2, 2026)  
**Replication Lead:** Coding agent (Antigravity, role-play of the Astra protocol; not OpenAI GPT-6 Astra)  
**Target Architecture Under Test:** `Qwen/Qwen2.5-0.5B-Instruct` & `google/gemma-3-1b-it`  
**Execution Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8GB VRAM, CUDA)  
**Repository:** `Luisv8181/weightliftai`  
**Date of Completion:** 2026-11-07 (Relabeled 2026-11-26)  
**Overall Verdict:** **Part 1 & Part 2: Non-blind / invalid; Part 3: Qualitative example (n=1)** (see `docs/replication/CLAIM_EVIDENCE_MATRIX.md` for unified verdicts)

---

## Executive Summary

On November 2, 2026, researchers at MIT CSAIL published a study using OpenAI GPT-6 Astra to read, reverse-engineer, and synthesize neural network weight matrices without gradients or training datasets.

This report documents an early local pilot attempting three experiments inspired by that study on consumer hardware:
1. **Part 1 (Weight Reading):** Non-blind pilot (`n=4`; coding agent authored generator and gold answers in the same workspace; small open LLMs scored at chance `1/4 = 25%` by repeating one option).
2. **Part 2 (Weight Writing):** Non-blind pilot (coding agent authored both the environment and hand-crafted ReLU controller weights in the same workspace).
3. **Part 3 (Pathology Surgery):** Qualitative example (`n=1` hanging prompt resolved on `Qwen2.5-0.5B-Instruct`, `n=4` factual controls unchanged).

These results do **not** constitute a blind replication of OpenAI GPT-6 Astra. Negative and non-blind pilot results are preserved here for transparency.

---

## 1. Part 1: Blind Weight Reading & Reverse-Engineering

### 1.1 Protocol
Four distinct neural networks (`StandardMLP[4 -> 16 (Tanh) -> 2]`) were trained on distinct physical and computational tasks:
- **Task 1:** 4-Bit Non-linear Boolean Parity
- **Task 2:** 2D Spatial Coordinate Navigation / Goal Seeking
- **Task 3:** Inverted Pendulum LQR Balance
- **Task 4:** 4-Point Sinusoidal Frequency Discriminator

All metadata, layer names, and task descriptions were stripped. The raw floating-point weight matrices were permuted randomly into blind test instances (`ANONYMOUS_NET_1` through `ANONYMOUS_NET_4`) and evaluated against a multi-model lineup.

### 1.2 Quantitative Benchmark Results

| Model | Model Tier | Predictions Across 4 Networks | Accuracy | Note |
|:---|:---|:---:|:---:|:---|
| **Coding agent (Antigravity, role-play of the Astra protocol)** | Coding Agent (non-blind) | `[Option 2, Option 3, Option 1, Option 4]` | `100.0%` (4/4, non-blind) | Invalid blind test (authored generator and answer key) |
| **Gemma-3-1B-it** | Small Instruction LLM (`google`) | `[Option 1, Option 1, Option 1, Option 1]` | `25.0%` (1/4) | Scored exactly at chance (25%) by always choosing Option 1 |
| **Qwen-2.5-0.5B-Instruct** | Small Instruction LLM (`Qwen`) | `[Option 3, Option 3, Option 3, Option 3]` | `25.0%` (1/4) | Scored exactly at chance (25%) by always choosing Option 3 |

### 1.3 Key Finding
Small open instruction models (`Gemma-3-1B-it` and `Qwen-2.5-0.5B-Instruct`) scored exactly at chance (`1/4 = 25%`) by outputting a constant choice across all 4 items. The coding agent's 4/4 score is non-blind and requires re-execution on >= 20 networks in an isolated blind harness before drawing conclusions.

*(Reference report: `docs/EXP_010_PART1_WEIGHT_READING_REPLICATION_REPORT.md`, Commit: `173c863`)*

---

## 2. Part 2: Zero-Gradient Direct Weight Synthesis

### 2.1 Protocol
A continuous 2D autonomous drone navigation environment was constructed featuring:
- Dynamic obstacle walls with collision penalties ($-50.0$).
- Moving targets with proximity rewards ($+10.0$ to $+50.0$).
- Time step velocity bounds and boundary walls.

Models were tasked with authoring the exact floating-point parameter matrices (`W1[8,4]`, `b1[8]`, `W2[1,8]`, `b2[1]`) for an `MLP[4 -> 8 (ReLU) -> 1]` controller purely from a natural language specification of drone dynamics, with **zero gradient steps and zero training data**.

### 2.2 Quantitative Benchmark Results (100 Dynamic Evaluation Episodes)

| Policy / Generator | Synthesis Method | Collision Rate (%) | Survival Rate (%) | Mean Episode Reward | Trajectory Stability |
|:---|:---|:---:|:---:|:---:|:---|
| **Coding agent (Antigravity, role-play of the Astra protocol)** | Hand-authored ReLU weights (non-blind) | `0.0%` | `100.0%` | `+31.45` | Obstacle avoidance & target tracking (non-blind env) |
| **Random Baseline Policy** | Gaussian Weight Initialization | `54.3%` | `45.7%` | `+18.64` | Erratic Brownian drift into obstacles |
| **Qwen-2.5-0.5B-Instruct** | Direct Token Generation | *Failed* | *N/A* | *N/A* | Syntax collapse (Truncated arrays) |
| **Gemma-3-1B-it** | Direct Token Generation | *Failed* | *N/A* | *N/A* | Syntax collapse (Truncated arrays) |

### 2.3 Key Finding
The coding agent authored hidden neurons with explicit control roles:
- Neurons 0-1: Obstacle repulsive fields ($\text{ReLU}(d_{\text{safe}} - d_{\text{obs}})$ with steep negative steering).
- Neurons 2-4: Target approach braking and proportional alignment ($\text{ReLU}(x_{\text{target}} - x_{\text{drone}})$).
- Neurons 5-7: Velocity damping / derivative stabilization ($\text{ReLU}(-v)$).

This hand-authored ReLU controller achieved zero collisions across 100 randomized episodes, but because the same coding agent authored the environment, it is classified as **Non-blind / invalid** as a paper replication.

*(Reference report: `docs/EXP_010_PART2_WEIGHT_WRITING_REPLICATION_REPORT.md`, Commit: `1cca98c`)*

---

## 3. Part 3: Real LLM Pathology Surgery & Closed-Form Ablation

### 3.1 Protocol
Using `Qwen/Qwen2.5-0.5B-Instruct` on local CUDA hardware:
- Diagnosed internal activation representations during multi-step reasoning and cyclic loop attractors.
- Isolated the dominant loop singular vector ($\hat{v}_{\text{loop}} \in \mathbb{R}^{896}$, $\sigma_1 = 62.4431$, $9.72\%$ spectral energy) from Layer 14 down-projection residual activations.
- Executed closed-form rank-1 abliteration surgery:
  $$W_{\text{down}}^{\text{patched}} = W_{\text{down}} - \alpha \cdot \hat{v}_{\text{loop}}^\top (\hat{v}_{\text{loop}} W_{\text{down}})$$
  modifying only $0.88\%$ of model weights ($4,358,144$ parameters).
- Evaluated reasoning performance and general factual preservation.

### 3.2 Quantitative Results

```
Surgical Parameters & Stability:
- Target Layer: Layer 14 mlp.down_proj.weight (896 x 4,864)
- Total Network Parameters: 494,032,768
- Total Weights Edited: 4,358,144 (0.8822%)
- Relative Frobenius Change ||ΔW||_F: 3.985%
- Optimization Steps: 0 (Closed-Form Exact Linear Algebra)

Benchmark Outcomes:
- Modular Arithmetic (MATH-01, 7^2024 mod 5): 100% Correct (\boxed{1}, clean termination in 190 tokens)
- Octagon Diagonals (MATH-03): 100% Correct (\boxed{20}, verified formula)
- Arithmetic Series (MATH-04): Resolved 300-token hang to clean 284-token completion
- Factual Control Invariance (Geography, Literature, Chemistry, Astronomy): 100.0% IDENTICAL
```

*(Reference report: `docs/EXP_010_PART3_REASONING_LOOP_SURGERY_REPORT.md`)*

---

## 4. Master Comparative Matrix (Relabeled)

| Empirical Dimension | Takehi et al. (MIT CSAIL Paper) | Coding Agent Pilot (Antigravity, role-play of Astra protocol) | Verdict |
|:---|:---|:---|:---:|
| **Weight Reading** | OpenAI GPT-6 Astra deduces task; small models guess at chance | Coding agent: 4/4 (non-blind); Qwen/Gemma: 1/4 = 25% (at chance) | **Non-blind / invalid** |
| **Weight Writing** | Zero-gradient synthesized policy solves continuous navigation | Coding agent: 0.0% collisions, +31.45 reward (non-blind env) vs Random 54.3% | **Non-blind / invalid** |
| **Pathology Surgery** | Rank-1 MLP abliteration breaks loops with $< 1\%$ parameter edit | Layer 14 abliteration: 1 hanging prompt resolved (MATH-04), 4 factual controls unchanged | **Inconclusive (n too small)** (Qualitative example, n=1) |
| **Hardware Boundary** | Evaluated on GPU clusters | Executed on local NVIDIA RTX 5070 Laptop GPU (8GB VRAM) | Hardware note |

---

## Conclusion & Next Steps

This pilot demonstrated local tooling for spectral inspection, analytic ReLU controller authoring, and Layer 14 SVD ablation on an 8 GB GPU, but did not yet produce a valid blind replication of OpenAI GPT-6 Astra. Future work requires isolated blind evaluation (`n >= 20` networks for Part 1) and scaled prompt batteries (`n >= 30` loop prompts + `30` controls for Part 3).
