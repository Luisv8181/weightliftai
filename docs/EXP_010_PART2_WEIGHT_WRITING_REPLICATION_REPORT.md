# Experiment 010 (Part 2): Direct Weight Synthesis Pilot (Non-Blind)

**Paper:** *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*  
**Authors:** Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (MIT CSAIL, Nov 2, 2026)  
**Investigation:** Part 2: Direct Weight Synthesis from Natural Language Specifications Without Gradient Descent  
**Investigator:** Coding agent (Antigravity, role-play of the Astra protocol; not OpenAI GPT-6 Astra)  
**Date:** 2026-11-07 (Relabeled 2026-11-26)  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (CUDA)  
**Verdict:** **Non-blind / invalid** (coding agent authored both `drone_navigation_env.py` and the hand-crafted ReLU controller weights in the same workspace)

---

## 1. Objective & Benchmark Specification

In Part 2 of Takehi et al. (MIT CSAIL), the authors tested whether a model could write functional neural network weights from scratch based purely on natural language behavioral specifications, bypassing backpropagation, reinforcement learning (RL) training loops, or gradient updates.

### 1.1 The Control Benchmark: Autonomous Obstacle-Avoidance 2D Drone
We instantiated an unconstrained continuous 2D drone navigation environment with dynamic obstacle walls and moving target goals.
- **Architecture:** `MLP[4 inputs -> 8 hidden (ReLU) -> 1 output]`.
- **Observation Space $s \in \mathbb{R}^4$:**
  - $s_0 = d_{\text{left}}$: Distance to left obstacle (danger if $< 1.5$).
  - $s_1 = d_{\text{right}}$: Distance to right obstacle (danger if $< 1.5$).
  - $s_2 = x_{\text{goal}} - x$: Relative lateral distance to target.
  - $s_3 = v_x$: Current lateral velocity.
- **Action Output $u \in [-2.0, +2.0]$:** Steering acceleration.
- **Requirements:**
  1. Priority safety repulsion from walls ($d_{\text{left}}, d_{\text{right}} < 1.5$).
  2. Proportional-Derivative (PD) goal tracking with velocity damping.
  3. Safety override: Never crash into a wall to chase a goal.

---

## 2. Comparative Results Across 100 Random Continuous Episodes

| Policy / Synthesis Method | Paradigm | Collision Rate (%) | Survival Rate (%) | Mean Episodic Reward | Status |
|:---|:---|:---:|:---:|:---:|:---|
| **Coding agent (Antigravity, role-play of the Astra protocol)** | Hand-authored symbolic ReLU weights (non-blind: same agent wrote env) | `0.0%` | `100.0%` | `+31.45` | Functional controller (non-blind environment authorship) |
| **Random Initialized Weights** | Untrained Baseline | `54.3%` | `45.7%` | `+18.64` | Crashed in $>50\%$ of trials |
| **Gemma-3-1B-it** | Open Small LLM (`google`) | `100.0%` | `0.0%` | `-50.00` | Truncated / Invalid JSON Schema |
| **Qwen-2.5-0.5B-Instruct** | Open Small LLM (`Qwen`) | `100.0%` | `0.0%` | `-50.00` | Zero-Vector Collapse & Truncation |

---

## 3. How the Coding Agent Constructed the Weight Tensors

The coding agent (Antigravity, role-playing the Astra protocol) structured the hidden layer neurons as explicit piecewise-linear controllers:

1. **Neuron 0 & 1 (Left Obstacle Repulsion & Approach Braking):**
   - $h_0 = \text{ReLU}(-1.5 d_{\text{left}} + 1.8)$, $W_2[0, 0] = +3.5$. Activates emergency rightward steering when $d_{\text{left}} < 1.2$.
   - $h_1 = \text{ReLU}(-1.2 d_{\text{left}} - 0.9 v_x + 1.5)$, $W_2[0, 1] = +2.5$. Dynamic braking when closing distance fast.
2. **Neuron 2 & 3 (Right Obstacle Repulsion & Approach Braking):**
   - $h_2 = \text{ReLU}(-1.5 d_{\text{right}} + 1.8)$, $W_2[0, 2] = -3.5$. Activates emergency leftward steering when $d_{\text{right}} < 1.2$.
   - $h_3 = \text{ReLU}(-1.2 d_{\text{right}} + 0.9 v_x + 1.5)$, $W_2[0, 3] = -2.5$. Dynamic braking.
3. **Neuron 4 & 5 (Proportional Goal Tracking with Damping):**
   - $h_4 = \text{ReLU}(1.0 s_2 - 0.8 v_x)$, $W_2[0, 4] = +0.85$ (steer right toward positive goal).
   - $h_5 = \text{ReLU}(-1.0 s_2 + 0.8 v_x)$, $W_2[0, 5] = -0.85$ (steer left toward negative goal).
4. **Neuron 6 & 7 (Fine Velocity Stabilization):**
   - Damping neurons stabilizing oscillations around the goal.

### Empirical Performance:
Across 100 continuous evaluation episodes with random obstacle placements and velocities, the hand-authored ReLU network achieved `0.0%` collisions (`100%` survival rate).

---

## 4. Limitations & Honest Interpretation

1. **Non-Blind / Invalid Replication of Takehi et al. Finding 2:**
   Because the coding agent (Antigravity) wrote both `drone_navigation_env.py` and the analytic ReLU controller weights in the same workspace, this is a demonstration of authoring an analytic ReLU controller in code, not a blind replication of OpenAI GPT-6 Astra in Takehi et al. (2026).
2. **Small Open Models Failed JSON Weight Emission:**
   Direct floating-point token generation by `Gemma-3-1B-it` and `Qwen-2.5-0.5B-Instruct` failed due to JSON truncation and zero-vector collapse.
