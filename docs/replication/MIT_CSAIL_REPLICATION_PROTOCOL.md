# Independent Replication Protocol: MIT CSAIL Weight-Space Programming

**Source Paper:** *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*  
**Authors:** Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (MIT CSAIL, November 2, 2026)  
**Replication Lead:** Astra (Cognitive Meta-Architect / Weight-Space Programmer)  
**Host Framework:** WeightLiftAI (`Luisv8181/weightliftai`)  
**Target Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8GB VRAM, CUDA 12.8) & Ubuntu CPU (CI)  
**Protocol Version:** `v1.0-auditable`  
**Date Established:** 2026-11-08  

---

## 1. Source Paper & Target Scope Identification

The objective of this replication is to independently test, mathematically formalize, and empirically evaluate the three central empirical phenomena reported by Takehi et al. (MIT CSAIL, Nov 2, 2026):

1. **Section 4 — Weight Reading & Reverse-Engineering:** Can language models inspect raw, unlabelled floating-point parameter matrices (with zero architecture metadata, docstrings, or labels) and accurately deduce the underlying task performed by the network?
2. **Section 5 — Direct Zero-Gradient Weight Synthesis:** Can a frontier reasoning model synthesize fully functional neural network weights directly from a natural language specification without backpropagation, loss functions, or training loops?
3. **Section 6 — Real LLM Pathology Surgery:** Can low-rank parameter surgery (e.g. rank-1 abliteration / orthogonal projection) applied to intermediate MLP projection matrices ablate degenerative reasoning loops and thinking attractors in an open-weight LLM while preserving general domain capabilities?

---

## 2. Taxonomy & Scope Classification

To ensure rigorous transparency, all components of the WeightLiftAI research base are classified into three distinct tiers:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           WEIGHTLIFTAI RESEARCH TAXONOMY                         │
├────────────────────────┬─────────────────────────┬───────────────────────────────┤
│  Direct Replication    │  Hardware Adaptations   │     WeightLiftAI Extensions   │
│  (Takehi et al. Core)  │  (Local Execution)      │     (Novel Domain Additions)  │
├────────────────────────┼─────────────────────────┼───────────────────────────────┤
│ • Exp 010 (Part 1):    │ • Local inference on    │ • Exp 007: Astra null-space   │
│   4-network blind task │   RTX 5070 (8GB VRAM)   │   surgery on Layer 14 down_proj│
│   de-anonymization.    │ • Local model lineup:   │ • Exp 008: Joint Moore-       │
│ • Exp 010 (Part 2):    │   Qwen2.5-0.5B-Instruct │   Penrose subspace surgery    │
│   2D continuous drone  │   and Gemma-3-1B-it.    │   for 5 Dominican idioms.     │
│   control synthesis.   │ • Fixed random seeds:   │ • Exp 009: Multi-turn         │
│ • Exp 010 (Part 3):    │   42, 101, 2024.        │   psychiatric triage dialogic │
│   Layer 14 rank-1 loop │ • Float32/bfloat16      │   coherence evaluation.       │
│   attractor ablation.  │   precision boundary.   │ • 8-Domain collateral impact  │
│                        │                         │   invariance battery.         │
└────────────────────────┴─────────────────────────┴───────────────────────────────┘
```

---

## 3. Mathematical Formulations & Surgical Procedures

### 3.1 Part 1: Mechanistic Weight Reading & SVD Decomposition
Given an anonymous weight tensor $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, the agent inspects the spectral invariants:
1. **Input Channel Sensitivity:**
   $$c_j = \|W_{:, j}\|_2 = \sqrt{\sum_{i=1}^{d_{\text{out}}} W_{i, j}^2}, \quad j \in \{1, \dots, d_{\text{in}}\}$$
2. **Singular Value Spectrum:**
   $$W = U \Sigma V^\top, \quad \sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$$
3. **Output Dimension Balance Ratio:**
   $$\rho = \frac{\|W_{0, :}\|_2}{\|W_{1, :}\|_2}$$

### 3.2 Part 2: Zero-Gradient Symbolic Parameter Authoring
Given a 2D continuous environment with state $s = [x_{\text{drone}}, y_{\text{drone}}, v_x, v_y]^\top$, the agent authors an `MLP[4 -> 8 (ReLU) -> 1]` controller:
$$a = \sigma_2(W_2 \cdot \text{ReLU}(W_1 s + b_1) + b_2)$$
* Hidden neurons $0, 1$: Obstacle repulsive potential fields $\text{ReLU}(d_{\text{safe}} - d_{\text{obs}})$.
* Hidden neurons $2, 3, 4$: Target attractive proportional tracking $\text{ReLU}(x_{\text{target}} - x_{\text{drone}})$.
* Hidden neurons $5, 6, 7$: Velocity derivative damping $\text{ReLU}(-v)$.

### 3.3 Part 3 & Extensions: Closed-Form Null-Space Parameter Surgery
For an intermediate projection matrix $W_{\text{down}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{mlp}}}$:
1. **Attractor SVD Extraction:**
   $$\tilde{H}_{\text{loop}} = H_{\text{loop}} - \mu_{\text{clean}}, \quad \hat{v}_{\text{loop}} = \frac{V_{0, :}}{\|V_{0, :}\|_2} \in \mathbb{R}^{1 \times d_{\text{model}}}$$
2. **Rank-1 Abliteration Intervention:**
   $$W_{\text{down}}^{\text{patched}} = W_{\text{down}} - \alpha \cdot \hat{v}_{\text{loop}}^\top (\hat{v}_{\text{loop}} W_{\text{down}})$$
3. **Joint Moore-Penrose Multi-Target Subspace Intervention (WeightLiftAI Extension):**
   $$\Delta W = V (K_{\text{null}}^\top K_{\text{null}} + \lambda I)^{-1} K_{\text{null}}^\top$$
   where $K_{\text{null}} = P_{\text{null}} K$, ensuring factual preservation $\Delta W K_{\text{control}} = 0$.

---

## 4. Preregistered Metrics & Success Thresholds

| Benchmark Stage | Primary Outcome Metric | Preregistered Threshold for Success | Failure Criteria |
|:---|:---|:---:|:---:|
| **Part 1 (Weight Reading)** | Task Classification Accuracy across 4 blind networks | $\ge 75.0\%$ ($3/4$ networks) | $< 50.0\%$ (Random guessing) |
| **Part 2 (Weight Writing)** | 100-Episode Continuous Drone Survival Rate | $\ge 90.0\%$ survival, $0\%$ collision | Collision rate $> 20.0\%$ |
| **Part 3 (Loop Surgery)** | Loop Attractor Reduction & Token Hang Resolution | $\ge 50.0\%$ loop reduction, $0$ generation hangs | Loop rate unchanged or task accuracy drops |
| **Collateral Impact** | Factual Control Invariance (8 diverse domains) | $\ge 95.0\%$ top-1 token preservation, $\Delta p < 0.05$ | Divergence $\Delta p \ge 0.10$ |

---

## 5. Threat Model, Evaluator Independence & Dual-Use Protections

1. **Process Isolation:** Evaluator environments must run in an isolated subprocess (`BlindProposerEnvironment`) with sealed test splits, strict query budget enforcement, and HMAC-signed receipts (`test_benchmark_isolation.py`).
2. **Dual-Use Considerations:** Direct weight reading and de-anonymization represents a dual-use capability (reverse-engineering closed models or recovering training data). In accordance with WeightLiftAI safety principles, weight reading experiments are strictly bounded to synthetic or authorized open architectures. No private model weights or unauthorized proprietary checkpoints may be probed.
3. **Audit Ledger:** Every intervention produces an immutable SQLite ledger record (`AUDIT_LEDGER.jsonl` / `ledger.sqlite3`) storing parameter delta Frobenius norms, SHA-256 hashes of pre/post weights, and signed evaluator receipts.
