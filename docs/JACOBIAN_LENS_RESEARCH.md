# Jacobian Lens and J-Space Research Module

**Status:** Research design / implementation specification (not yet an implemented capability)

**Tracking:** [Issue #6](https://github.com/Luisv8181/weightliftai/issues/6)

## Why this belongs in WeightLiftAI

WeightLiftAI currently investigates model behavior through parameters, activations, SAE features, and controlled interventions. A Jacobian Lens adds a complementary question: which directions in an internal activation are positioned to influence what the model could verbalize later?

This is distinct from weight space. Model weights determine the transformations; the Jacobian Lens analyzes how activation perturbations propagate through those transformations toward output logits. A J-space-like analysis may therefore connect representation cartography to causal behavior engineering without assuming that editing weights is the right intervention.

## Research basis

Anthropic's 2026 work introduces the Jacobian lens (J-lens). For a residual-stream activation at layer `l` and source position `t`, the method studies how perturbations affect later residual states and output logits. The paper averages Jacobian effects across positions and contexts to reduce prompt-specific artifacts. The resulting token-associated directions are read out through the model's unembedding. The authors define J-space using sparse, nonnegative combinations of J-lens directions.

Primary references:

- [Anthropic research overview: A global workspace in language models](https://www.anthropic.com/research/global-workspace)
- [Gurnee et al. (2026): Verbalizable Representations Form a Global Workspace in Language Models](https://arxiv.org/abs/2607.15495)
- [Full paper and methods](https://transformer-circuits.pub/2026/workspace/index.html)
- [Ostrow et al. (2026): A Mathematical Theory of the Jacobian Lens](https://arxiv.org/abs/2610.31955)

These results are model- and method-specific. WeightLiftAI must test whether analogous structures are detectable in its supported open-weight models. It must not assume that Claude's findings transfer unchanged.

## Mathematical sketch

Let `h[l,t]` be the residual-stream activation at layer `l` and token position `t`. A local perturbation propagates to later residual states according to a Jacobian such as:

`d h[L,t'] / d h[l,t]`, for `t' >= t`.

A practical context-averaged map can be estimated as:

`J_l = E_context, t, t' [ d h[L,t'] / d h[l,t] ]`.

A vocabulary readout then composes this map with the model's final normalization and unembedding. In practice, implementations may need to estimate selected rows, vector-Jacobian products, or low-rank approximations rather than materialize a prohibitively large full Jacobian.

The paper's J-space is not simply the span of a few vectors. The vocabulary-associated directions are overcomplete; the authors operationalize J-space through sparse nonnegative combinations, with results depending on the sparsity choice. WeightLiftAI should expose that choice and report sensitivity rather than hide it behind a single visualization.

## Proposed package boundaries

The first implementation should be analysis-only and architecture-aware:

- `jacobian_lens.py`: interfaces and validated computation primitives for selected layers, positions, and output horizons.
- `context_aggregation.py`: deterministic aggregation across a frozen prompt corpus, with corpus hashes and sample counts.
- `readout.py`: token-ranked readouts with token IDs, decoded strings, normalization settings, and layer/position metadata.
- `jspace.py`: sparse candidate extraction, top-k alternatives, and sensitivity analysis over sparsity and normalization.
- `diagnostics.py`: logit-lens comparison, context-shuffle controls, random-direction baselines, rank stability, and finite-difference checks.
- `artifacts.py`: compact, hashed artifacts integrated with the experiment registry; raw activations remain local by default.

The package should not assume every Hugging Face architecture exposes the same residual stream or hook points. Model-family adapters must declare exactly which tensor is being analyzed and how it maps to the final residual stream.

## Compute strategy for consumer hardware

A full Jacobian across every layer, token, future position, and vocabulary is likely too expensive for a consumer GPU. Build in escalating modes:

1. **Sanity mode:** tiny synthetic transformer; compare autograd results with finite differences.
2. **Single-context mode:** selected layer, position, and future-token horizon; label as prompt-specific.
3. **Small-corpus mode:** aggregate a modest, frozen set of contexts; report standard errors or bootstrap intervals where meaningful.
4. **Approximation mode:** optional low-rank or sampled-output approximations, always labeled and compared with an exact small-model reference.
5. **Scale-up mode:** only after profiling memory, runtime, and numerical stability.

Use `torch.autograd.grad` / vector-Jacobian products where possible. Avoid retaining computation graphs across unrelated prompts. Record dtype, precision, device, batch size, sequence length, truncation policy, and any approximation parameters. Do not silently substitute a gradient of a single current-token logit for the paper's context-averaged future-output construction.

## Validation ladder

### 1. Numerical correctness

- Unit-test Jacobian-vector and vector-Jacobian products on tiny deterministic networks.
- Compare autograd estimates with finite differences over several epsilon values.
- Test shape, device, dtype, and gradient-cleanup behavior.
- Verify model parameters are unchanged after analysis.

### 2. Readout validity

- Compare J-lens readouts with a conventional logit lens at matched layers and positions.
- Compare single-context and context-averaged results separately.
- Measure rank stability across contexts and held-out prompts.
- Include shuffled-context, shuffled-label, and random-direction controls.
- Track tokenizer effects; a token direction is not automatically a complete multi-token concept.

### 3. Candidate J-space structure

- Report active token directions and sparsity assumptions.
- Test multiple `k` values and normalization choices.
- Quantify stability across prompt samples, seeds, and model revisions.
- Avoid claiming that a 2D projection is the literal geometry of a model's internal workspace.

### 4. Causal tests

Only after readout validity is established, test reversible activation interventions:

- Baseline continuation.
- Candidate-direction addition, subtraction, or projection swap.
- Equal-norm sham intervention.
- Random direction matched for norm and layer.
- Held-out target and unrelated regression tasks.

Freeze the hypothesis, dataset split, primary metric, and regression thresholds before intervention. Report all attempts, including null and harmful results. A change in verbal report alone is not sufficient evidence of broad task improvement.

## Initial pilot design

Start with a small inspectable causal language model that fits local hardware. Use two deliberately narrow tasks:

1. **Single-token concept report:** ask the model to select an item from a fixed category and later report it. Test whether the J-lens identifies the selected item before the report.
2. **Contextual reasoning:** use short, deterministic tasks where intermediate concepts are specified in advance and can be evaluated independently.

After the numerical and readout checks pass, add matched English/Spanish prompts. Evaluate tokenization, lexical confounds, and language-specific stability. Do not infer cultural understanding from token rankings or dialect labels alone.

## Experiment registry requirements

Every run should record:

- Model and tokenizer IDs plus immutable revisions.
- Architecture adapter and exact residual tensor definition.
- Prompt-corpus ID, version, split, license/provenance, and content hash.
- Source layer, source token positions, future-output horizon, and aggregation rule.
- Jacobian method, exact or approximate status, dtype, precision, batch size, and memory/runtime.
- Normalization and unembedding details; tokenizer vocabulary mapping.
- Sparsity method, `k`, thresholds, and candidate directions.
- Random seeds, code commit, environment, hardware, and artifact hashes.
- Baseline, sham, random-control, intervention, and regression metrics.

Keep private prompts and raw activation tensors local unless publication is explicitly reviewed and approved.

## Non-claims and safeguards

- J-space is not weight space and is not a separate physical model layer.
- A J-lens readout is not a complete transcript of hidden reasoning.
- A high-ranked token is not proof of a stable semantic concept.
- Decodability or correlation does not establish causal importance.
- Causal influence on a verbal report does not automatically establish improvement on a broader task.
- The global-workspace analogy is functional and does not establish phenomenal consciousness.
- Do not assume the method transfers from Claude to Gemma, Qwen, or other open-weight models.
- Do not publish sensitive prompts, private activations, or identifiable data.

## Delivery sequence

- [ ] Add architecture adapter contract and tiny-model numerical tests.
- [ ] Implement single-context Jacobian readout for one supported model family.
- [ ] Add context aggregation and reproducibility metadata.
- [ ] Add logit-lens comparison and shuffled/random controls.
- [ ] Add sparse candidate extraction with sensitivity reporting.
- [ ] Add compact visualization to Representation Cartography.
- [ ] Integrate results with the experiment registry.
- [ ] Run the single-token concept-report pilot.
- [ ] Add reversible causal intervention experiments only after readout validation.
