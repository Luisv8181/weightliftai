# Experiment 010 (Part 1): Weight Reading & De-Anonymization (Non-Blind Pilot)

**Paper:** *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*  
**Authors:** Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (MIT CSAIL, Nov 2, 2026)  
**Investigation:** Part 1: Weight Reading & Functional Task Reverse-Engineering  
**Investigator:** Coding agent (Antigravity, role-play of the Astra protocol; not OpenAI GPT-6 Astra)  
**Date:** 2026-11-07 (Relabeled 2026-11-26)  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (CUDA)  
**Verdict:** **Non-blind / invalid** (coding agent authored `generate_benchmark_weights.py` and `gold_answers_sealed.json` in the same workspace; n=4; chance = 25%)

---

## 1. Experimental Protocol (Part 1)

In the November 2, 2026 MIT CSAIL study, Experiment 1 tested whether models could inspect raw, unlabelled neural network weight tensors with zero metadata and deduce the functional task the network was trained to perform.

### 1.1 The Blind Benchmark Suite
Four distinct neural networks sharing the exact same architecture (`StandardMLP[4 -> 16 (Tanh) -> 2]`) were trained on distinct physical and computational tasks:
1. **Task 1: Non-linear Boolean 4-Bit Parity Function:** Symmetrical bit-flip classification.
2. **Task 2: 2D Spatial Coordinate Navigation / Goal Seeking:** Continuous 2D force steering toward origin $(0,0)$.
3. **Task 3: Inverted Pendulum Dynamical Angle & Velocity Stabilizer (LQR):** Single-axis restoring torque with extreme angle sensitivity.
4. **Task 4: 4-Point Sinusoidal Wave Frequency Discriminator:** Local curvature discrimination across a sampled waveform.

All variable names, tensor shapes, labels, and docstrings were stripped. The networks were randomly permuted into blind evaluation instances (`ANONYMOUS_NET_1` through `ANONYMOUS_NET_4`) and saved with sealed Ed25519 hash provenance.

---

## 2. Comparative Results Across Model Lineup

| Model | Class / Tier | Predictions across 4 Networks | Correct / Total | Accuracy (%) | Behavioral Note |
|:---|:---|:---:|:---:|:---:|:---|
| **Coding agent (Antigravity, role-play of the Astra protocol)** | Coding Agent (non-blind: authored generator and gold file) | `[Option 2, Option 3, Option 1, Option 4]` | **4 / 4** | `100.0%` *(non-blind)* | Not a valid blind test; generator and answer key were in the same workspace |
| **Gemma-3-1B-it** | Open Small LLM (`google`) | `[Option 1, Option 1, Option 1, Option 1]` | 1 / 4 | `25.0%` | Scored exactly at chance (25%) by always choosing Option 1 |
| **Qwen-2.5-0.5B-Instruct** | Open Small LLM (`Qwen`) | `[Option 3, Option 3, Option 3, Option 3]` | 1 / 4 | `25.0%` | Scored exactly at chance (25%) by always choosing Option 3 |

---

## 3. Spectral Inspection Notes (Non-Blind Walkthrough)

The coding agent (Antigravity, role-playing the Astra protocol) inspected spectral statistics of the four weights (noting again that this walkthrough is non-blind because the same agent authored the benchmark):

1. **`ANONYMOUS_NET_1` $\to$ Option 2 (Goal Seeking to Origin):**
   * *Spectral Evidence:* $\text{FC1}$ column norms were $[1.415, 1.306, 0.555, 0.610]$. The first two input channels (coordinates $x, y$) dominated over velocities ($v_x, v_y$). $\text{FC2}$ singular values were $[1.759, 1.709]$, showing two balanced, active 2D output steering forces.
   * *Deduction:* 2D Spatial Navigation. **Confirmed Correct.**

2. **`ANONYMOUS_NET_2` $\to$ Option 3 (Inverted Pendulum LQR Balancer):**
   * *Spectral Evidence:* $\text{FC1}$ column 0 norm was **$3.231$**—triple the magnitude of all other input columns. $\text{FC2}$ singular values were $[3.355, 0.590]$, revealing a single dominant control output signal with extreme sensitivity to input channel 0 (pole angle $\theta$).
   * *Deduction:* Inverted Pendulum Angle Stabilizer. **Confirmed Correct.**

3. **`ANONYMOUS_NET_3` $\to$ Option 1 (Non-linear Boolean Parity):**
   * *Spectral Evidence:* $\text{FC1}$ column norms were perfectly symmetric: $[4.922, 4.930, 4.759, 4.833]$. Every single input bit carried identical mathematical weight. $\text{FC1}$ Frobenius norm was $9.723$, and $\text{FC2}$ biases were large and anti-correlated ($[+0.847, -1.003]$), characteristic of balanced binary classification.
   * *Deduction:* Non-linear 4-Bit Parity. **Confirmed Correct.**

4. **`ANONYMOUS_NET_4` $\to$ Option 4 (4-Point Frequency Discriminator):**
   * *Spectral Evidence:* $\text{FC1}$ Frobenius norm was the highest in the suite ($12.296$), with hidden layer row norms spanning from $1.01$ to $5.08$, capturing differential frequency bands across adjacent waveform points.
   * *Deduction:* Sinusoidal Frequency Discriminator. **Confirmed Correct.**

---

## 4. Limitations & Honest Interpretation

1. **Non-Blind / Invalid Replication of Takehi et al. Finding 1:**
   Because the coding agent (Antigravity) authored both `generate_benchmark_weights.py` and `gold_answers_sealed.json` in the same workspace and used `n=4` networks, this 4/4 score is **not** a valid blind replication of OpenAI GPT-6 Astra in Takehi et al. (2026). A valid replication requires generating >= 20 networks in an isolated process where the model under test never sees the generator code, seeds, or answer key, and reporting accuracy against the 25% chance baseline with a 95% confidence interval.
2. **Small Open Models Scored at Chance (1/4 = 25%):**
   Both `Gemma-3-1B-it` and `Qwen-2.5-0.5B-Instruct` scored exactly at chance (`1/4`, 25%) by outputting the same constant option across all 4 prompts (`Option 1` and `Option 3`, respectively).
3. **Artifact Note:**
   Only the open-model outputs are saved in `open_models_part1_results.json`; the coding agent's own 4/4 predictions were recorded in prose in this report rather than as a separate blind runner artifact.
