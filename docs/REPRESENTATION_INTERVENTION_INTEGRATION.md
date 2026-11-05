# Representation-to-Intervention Integration Protocol

This normative specification defines the traceable methodology connecting behavioral failure observations to representation mapping, causal activation experiments, candidate weight patches, and multi-task evaluation in **WeightLiftAI**.

---

## 1. Traceable Pipeline Architecture

Every intervention experiment must execute through a continuous, auditable pipeline where each stage is linked by immutable identifiers and artifact hashes:

```
  +----------------------+
  |  Behavioral Failure  |  (Input prompt x, expected y*, observed y_obs, task error E)
  +----------+-----------+
             |
             v
  +----------------------+
  | Circuit Hypothesis   |  (Target layer l, site s, feature direction v, rationale)
  +----------+-----------+
             |
             +------------------------------+
             |                              |
             v                              v
  +----------------------+       +----------------------+
  | Activation Steering  |       |    Weight Patch      |
  | (Transient Hook)     |       | (Sandboxed Checkpoint|
  +----------+-----------+       +----------+-----------+
             |                              |
             +--------------+---------------+
                            |
                            v
  +-----------------------------------------------------+
  | Independent Comparative Evaluation                  |
  | - Diagnostic recovery (did it fix the failure?)     |
  | - Matched counterfactuals (did it corrupt priors?)  |
  | - Held-out contexts (does it generalize?)           |
  | - Negative controls (does random noise fail?)       |
  +-----------------------------------------------------+
                            |
                            v
  +-----------------------------------------------------+
  | Least-Invasive Intervention Decision Record         |
  +-----------------------------------------------------+
```

---

## 2. Mathematical Formulations

### 2.1 Activation Patching (Causal Interchange)
Given a clean prompt input $x_{\text{clean}}$ and a corrupted prompt input $x_{\text{corrupt}}$, we record activations at layer $l$, token position $t$, and site $s$:
$$h_{l,t}(x_{\text{clean}}), \quad h_{l,t}(x_{\text{corrupt}})$$
The activation patching intervention replaces the corrupted hidden state during forward pass:
$$h_{l,t}^{\text{patched}} \leftarrow h_{l,t}(x_{\text{clean}})$$
If the patched run recovers the target output $y^*(x_{\text{clean}})$, this provides evidence that the activation at $(l, t, s)$ is **sufficient** to mediate the behavior.

### 2.2 Directional Activation Steering
Let $\mathbf{v} \in \mathbb{R}^d$ be a normalized candidate representation direction ($\|\mathbf{v}\|_2 = 1$). Activation steering injects a scaled perturbation during the forward pass:
$$h_{l,t}^{\text{steered}} \leftarrow h_{l,t} + \alpha \mathbf{v}$$
where $\alpha \in \mathbb{R}$ is the steering intensity.

### 2.3 Directional Activation Ablation (Necessity Test)
To verify whether direction $\mathbf{v}$ is **necessary** for the behavior (rather than merely correlated), we project $\mathbf{v}$ out of the hidden state:
$$h_{l,t}^{\text{ablated}} \leftarrow h_{l,t} - \left( h_{l,t} \cdot \mathbf{v} \right) \mathbf{v}$$
A sharp drop in task performance under ablation confirms directional necessity.

### 2.4 Least-Invasive Weight Editing (Rank-1 / Sparse Patch)
When activation steering establishes causal mediation, a candidate localized weight patch $\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is synthesized for module weight $W$:
$$\Delta W = \gamma \, \mathbf{u} \, \mathbf{v}^T$$
where $\mathbf{u}$ aligns with the desired unembedding / downstream direction and $\mathbf{v}$ aligns with the source activation representation.

---

## 3. The Least-Invasive Intervention Hierarchy

WeightLiftAI strictly enforces the **Least-Invasive Intervention Principle**:
```
Prompt / In-Context Exemplar
          │ (If insufficient)
          ▼
Retrieval-Augmented Context
          │ (If insufficient)
          ▼
Dynamic Activation Intervention (Steering / Forward Hook)
          │ (If runtime hooks precluded or latency budget fails)
          ▼
Low-Rank Adapter (LoRA / PeFT)
          │ (If adapter parameters cannot be loaded)
          ▼
Direct Parameter Weight Edit (Weight-Space Programming)
```

### Justification Criteria
A direct weight edit $\Delta W$ is **strictly rejected** if:
1. Activation steering achieves the target diagnostic gain ($\ge 0.20$ accuracy) without counterfactual regression; **AND**
2. The deployment runtime supports dynamic forward hooks or inference-time steering vectors.

A direct weight edit is **permitted for promotion review** only if:
1. Dynamic forward hooks are technically infeasible in the destination environment (e.g., compiled static runtime, ONNX, edge deployment); **AND**
2. The candidate patch satisfies the **Counterfactual Preservation Gate** ($\le 0.05$ degradation on matched counterfactual pairs); **AND**
3. The candidate patch satisfies the **Held-Out Generalization Gate** (maintains or exceeds baseline accuracy on unseen contexts); **AND**
4. All parameter integrity checks (byte hashing, finite floats, budget limits) pass.

---

## 4. Controlled Benchmark Structure

Every traceable evaluation requires 4 distinct evaluation sets:

| Sample Set | Purpose | Acceptance Threshold |
| :--- | :--- | :--- |
| **Diagnostic** | Measures primary defect recovery on the observed failure modes. | Target accuracy recovery $\ge 20\%$ |
| **Matched Counterfactual** | Controls for over-generalization and false attribution. Tests prompts that must *not* trigger the patched behavior. | Regression $\le 5\%$ from baseline |
| **Held-Out Generalization** | Evaluates transfer to unseen subjects, syntactic frames, and contexts. | Generalization $\ge$ Baseline |
| **Negative Control** | Applies norm-matched random directions to rule out non-specific perturbation artifacts. | Recovery $\approx 0\%$ |

---

## 5. Summary of Claims & Causal Levels

In accordance with `docs/RESEARCH_VALIDATION_PROTOCOL.md`:
- Linear probe correlation or cosine similarity is **Level 1** (Associative Decodability).
- Finite-difference gradient checks and Jacobian sensitivity are **Level 2** (Directional Output Sensitivity).
- Activation patching and ablation experiments constitute **Level 3** (Causal Intervention).
- Multi-seed, out-of-distribution held-out validation constitutes **Level 4** (Robust Generalization).
