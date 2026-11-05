# Jacobian Lens Method Specification & Validation Protocol

**Status:** Method Specification & Implementation Reference  
**Tracking:** [Issue #10 (Phase 2)](https://github.com/Luisv8181/weightliftai/issues/10)  
**References:**
- Anthropic (2026): *A global workspace in language models*
- Gurnee et al. (2026): *Verbalizable Representations Form a Global Workspace in Language Models* ([arXiv:2607.15495](https://arxiv.org/abs/2607.15495))
- Ostrow et al. (2026): *A Mathematical Theory of the Jacobian Lens* ([arXiv:2610.31955](https://arxiv.org/abs/2610.31955))

---

## 1. Mathematical Foundation

### 1.1 The Source Activation and Output Logit
Let a transformer decoder model process an input token sequence $x = (x_1, \dots, x_T)$.
- Let $h_{l,t} \in \mathbb{R}^{d}$ denote the residual-stream activation at layer $l \in \{0, \dots, L-1\}$ and source position $t \in \{1, \dots, T\}$.
- The forward computation through remaining layers $l+1, \dots, L$ produces the final residual state $h_{L, t'}$, which passes through final normalization $\text{Norm}(\cdot)$ (LayerNorm or RMSNorm) and the unembedding matrix $W_U \in \mathbb{R}^{V \times d}$:
  $$\mathbf{y}_{t'} = W_U \, \text{Norm}(h_{L, t'}) \in \mathbb{R}^{V}$$
  where $\mathbf{y}_{t'}[k]$ is the unnormalized logit for vocabulary token $k \in \{1, \dots, V\}$ at target position $t' \ge t$.

### 1.2 Vector-Jacobian Product (VJP)
For a specific target token $k$ at position $t'$, the directional sensitivity of the logit with respect to the intermediate representation $h_{l,t}$ is given by the gradient vector:
$$\mathbf{v}_{k}(l, t, t') = \nabla_{h_{l,t}} \mathbf{y}_{t'}[k] = \left( \frac{\partial \mathbf{y}_{t'}[k]}{\partial h_{l,t}} \right)^T \in \mathbb{R}^{d}$$

By the multivariate chain rule:
$$\mathbf{v}_{k}(l, t, t') = \left( \frac{\partial \mathbf{y}_{t'}[k]}{\partial h_{L, t'}} \cdot \frac{\partial h_{L, t'}}{\partial h_{l,t}} \right)^T = J_{l \to L}^T \nabla_{h_{L, t'}} \mathbf{y}_{t'}[k]$$
where $J_{l \to L} = \frac{\partial h_{L, t'}}{\partial h_{l,t}} \in \mathbb{R}^{d \times d}$ is the intermediate residual Jacobian.

### 1.3 Exact Normalization Handling
When backpropagating through `outputs.logits`, autograd automatically accounts for the derivative of the final normalization layer. For RMSNorm with weight $\gamma \in \mathbb{R}^d$:
$$\text{RMSNorm}(z) = \frac{z}{\text{RMS}(z)} \odot \gamma, \quad \text{RMS}(z) = \sqrt{\frac{1}{d} \sum_{i=1}^d z_i^2 + \epsilon}$$
The Jacobian of the normalization is:
$$\frac{\partial \text{RMSNorm}(z)}{\partial z} = \frac{1}{\text{RMS}(z)} \left( \text{diag}(\gamma) - \frac{\text{RMSNorm}(z) z^T}{d \cdot \text{RMS}(z)} \right)$$
Backpropagating directly from the scalar logit $\mathbf{y}_{t'}[k]$ through autograd naturally includes this normalization term, avoiding the scale distortion that occurs when projecting unnormalized hidden states.

### 1.4 Jacobian Lens Scoring vs. Logit Lens
- **Logit Lens:** Directly projects intermediate activation through the unembedding:
  $$\mathbf{s}_{\text{logit}}(h_{l,t}) = W_U \, \text{Norm}(h_{l,t}) \in \mathbb{R}^V$$
  *Implicit assumption:* Layers $l \to L$ act as an approximate identity map.
- **Jacobian Lens:** Measures the alignment between the activation and the output sensitivity direction:
  $$S_{\text{J-lens}}(h_{l,t}, k) = \langle \mathbf{v}_k(l, t, t'), \, h_{l,t} \rangle = \mathbf{v}_k(l, t, t')^T h_{l,t}$$
  *Meaning:* How much the current activation $h_{l,t}$ is actively driving the logit of token $k$ along the linearized path through the actual remaining layers.

---

## 2. Overcoming Shortlist Bias

A key methodological hazard identified in Issue #10 is **shortlist bias**:
If candidate tokens $k$ are restricted to the top-$k$ of the direct Logit Lens, the analysis can completely miss features that are transformed non-linearly across subsequent layers:
- If an activation represents a concept whose token expression has a near-zero direct logit projection at layer $l$, Logit Lens filtering will discard it.
- **Solution:** In WeightLiftAI, candidate evaluation supports:
  1. **Full-Vocabulary Readout:** For small vocabulary models or when evaluating specific target hypotheses.
  2. **Multi-Source Candidate Union:** Combining top-$k$ Logit Lens candidates, top-$k$ final model output predictions, and pre-registered target concept tokens.
  3. **Disclosed Shortlist Boundaries:** If top-$k$ filtering is used, the shortlist origin is recorded and labeled as a constrained candidate set.

---

## 3. Numerical Verification: Central Finite Differences

To prove mathematical correctness, every VJP computation is checked against numerical central finite differences:
$$\left[ \frac{\partial f}{\partial h_{l,t}} \right]_i \approx \frac{f(h_{l,t} + \epsilon \mathbf{e}_i) - f(h_{l,t} - \epsilon \mathbf{e}_i)}{2\epsilon}$$
where $\mathbf{e}_i$ is the $i$-th standard basis vector and $\epsilon = 10^{-4}$ (or dtype-appropriate value).

**Verification Requirements:**
- The numerical perturbation must be injected at the exact hooked module output.
- All attention masks, position IDs, and kwargs must be identical to the autograd forward pass.
- Maximum relative error must satisfy:
  $$\max_i \frac{2 \cdot |g_i^{\text{analytical}} - g_i^{\text{numerical}}|}{|g_i^{\text{analytical}}| + |g_i^{\text{numerical}}| + 10^{-7}} < 0.05$$

---

## 4. Controls & Baselines

A Jacobian direction cannot be interpreted in isolation without contrasting against null baselines:
1. **Norm-Matched Random Direction:** Compare cosine similarity against $\mathbf{r} \sim \mathcal{N}(0, I)$ scaled to $\|\mathbf{r}\|_2 = \|\mathbf{v}_k\|_2$.
2. **Context Aggregation:** Average $\mathbf{v}_k$ across a frozen prompt corpus to remove single-prompt surface artifacts:
   $$\bar{\mathbf{v}}_k = \frac{1}{N} \sum_{i=1}^N \mathbf{v}_k^{(i)}$$
   Report the corpus hash and standard error across contexts.
3. **Layer Controls:** Compare early, middle, and late layers to trace where directional alignment with final output tokens emerges.
