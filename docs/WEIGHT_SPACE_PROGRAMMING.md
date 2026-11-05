# Weight-Space Programming and Continuous Model Improvement

**Status:** Research and architecture specification; not an implemented capability.

**Tracking:** [Issue #8](https://github.com/Luisv8181/weightliftai/issues/8)

## Thesis

WeightLiftAI should investigate a workflow in which a language model can inspect a neural network's structure and selected parameter information, reason about a narrowly defined failure, propose a bounded repair, and help generate small specialist networks. This is a form of **Weight-Space Programming (WSP)**: natural-language-guided diagnosis and modification of model parameters or architecture, followed by deterministic verification.

The supplied video screenshots describe early experiments in which LLMs read network weights, repair small networks, and generate networks from scratch. These are promising research leads, not proof of general-purpose weight-editing ability. WeightLiftAI should reproduce them independently and report the limits.

## What we are trying to build

A local-first model engineering workbench with a small specialist optimizer that can:

- Read architecture metadata and selected tensor summaries.
- Diagnose controlled defects and recurring evaluation failures.
- Propose a hypothesis and an explicit, bounded patch.
- Generate tiny networks for well-specified tasks.
- Inspect previous experiment outcomes and propose the next experiment.
- Track candidate model versions and recommend whether to retain, reject, or roll back a change.

The specialist does not need to be a general-purpose reasoning champion. It needs to be reliable at a constrained engineering workflow. A general LLM, a small local specialist, and deterministic tooling should be compared rather than assuming the specialist is automatically better.

## System architecture

### 1. Read-only model inspector

Expose model configuration, tensor names, shapes, dtypes, parameter counts, checksums, and selected statistics. Access to raw tensors should be scoped and read-only by default. Tensor metadata alone cannot reveal the full function of a weight; diagnosis may also require activations, training context, and behavioral tests.

### 2. Failure and evidence collector

Collect sanitized, consented, versioned failure cases. Store the source model revision, task, prompt or input provenance, expected behavior, observed behavior, and split assignment. Keep private examples and raw activations local by default.

### 3. Specialist planner

Convert a failure into a typed proposal containing:

- A falsifiable hypothesis.
- Candidate tensors, modules, or architecture components.
- Proposed intervention and expected effect.
- Parameter/tensor allowlist and maximum delta budget.
- Frozen primary metric, regression thresholds, and controls.
- Resource estimate and uncertainty.

### 4. Patch generator

Generate a declarative patch or isolated model-definition code. Do not permit free-form unrestricted writes to the baseline checkpoint. Every patch must declare changed tensor names, shapes, element counts, norms, and hashes.

### 5. Sandboxed executor

Apply changes only to a disposable copy. Enforce allowlists, memory/runtime limits, parameter-delta budgets, timeouts, and no access to secrets. Preserve the original checkpoint as immutable.

### 6. Independent evaluator

Run target tests, held-out tests, negative controls, general capability checks, and distribution-shift evaluations. The optimizer cannot edit evaluation data, redefine success metrics after seeing results, or approve its own patch.

### 7. Promotion and rollback

A candidate becomes eligible for promotion only after independent evaluation and human approval. Package the base revision, patch/delta, hashes, evaluation report, known limitations, and rollback instructions.

### 8. Evidence-linked experiment memory

Retrieve prior runs to identify what has and has not worked. Record uncertainty and context. A finding from one model, layer, task, or dataset must not be treated as a universal rule.

## Research tracks

### A. Can an LLM read neural-network weights?

Start with tiny networks whose architecture and ground-truth parameters are known. Provide structured tensor summaries and test whether the model can identify injected defects. Compare against numerical diagnostics and random guessing. Establish what can be inferred from weights alone versus what requires activations or data.

### B. Can an LLM repair a known defect?

Corrupt selected weights in a tiny MLP trained on a controlled task. Ask the model to identify the defect and propose a patch. Compare with a no-op, random patch, hand-designed repair, and conventional gradient-based optimization. Measure both restoration of the original behavior and generalization.

### C. Can an LLM generate a network from scratch?

Use tiny deterministic tasks such as bounded addition or small classification. Require generated architecture and weights to be explicit, reproducible, and inspectable. Evaluate exhaustively over the declared input domain and separately on out-of-domain inputs. Compare against standard initialization and conventional training. A first-shot success on a toy task is not evidence of broad network synthesis capability.

### D. Can a small local specialist optimize models?

Benchmark a general-purpose LLM, a small local specialist, deterministic scripts, and a hybrid system on identical diagnosis and patch-generation tasks. Only train or fine-tune a specialist after a benchmark demonstrates a repeatable advantage worth the maintenance cost.

### E. Can the system improve models continuously?

The safe first version is continuous *monitoring and experiment proposal*, not continuous automatic weight mutation. The supervisor can identify recurring failures, retrieve evidence, propose experiments, and schedule periodic evaluation. Updates remain versioned, gated, and reversible. Never train directly on unreviewed production traffic or silently alter deployed weights.

## Determinism: where it belongs

A narrow model and constrained schemas may reduce variability, but they do not eliminate hallucinations or uncertainty. Deterministic code should handle exact arithmetic checks, tensor-shape validation, checksum verification, patch-budget enforcement, and acceptance gates. The language model should propose and explain; it should not be the final authority on correctness.

For example, if a tiny network is intended to add two bounded integers, an independent exact arithmetic evaluator should exhaustively test its declared domain. Do not rely on the generator's own explanation or sample demonstrations.

## Distribution shift and regression testing

Every proposed repair must be evaluated on:

- The original target failure and a held-out test split.
- Nearby inputs, paraphrases, and boundary cases.
- Relevant out-of-distribution or shifted inputs.
- Unrelated capabilities that must remain intact.
- Malformed or adversarial inputs where appropriate.
- Multiple seeds or replications where feasible.

Report target performance, regression rates, uncertainty/calibration where relevant, latency, memory, parameter-delta norms, and robustness. A narrow benchmark gain that causes broader regressions is not a successful repair.

## Intervention ladder

The platform should choose the least invasive evidence-supported fix:

1. Deterministic tool or guardrail for exact operations.
2. Prompt/context or retrieval change.
3. Inference-time activation intervention.
4. Adapter or LoRA.
5. Targeted weight-space patch.
6. Fine-tuning or architecture change.

Do not force a weight edit when a deterministic tool, better context, or runtime control solves the problem more reliably.

## Initial experiments

1. **Known-defect recovery:** inject a controlled corruption into a tiny MLP and test whether an LLM can locate and repair it.
2. **Addition micro-network:** generate a tiny network for a bounded addition domain; exhaustively test in-domain cases and separately test out-of-domain values.
3. **Patch integrity audit:** require a structured patch manifest; reject any undeclared tensor change or budget violation.
4. **Distribution-shift challenge:** evaluate each repair under shifted input distributions and unrelated tasks.
5. **Optimizer comparison:** compare general LLM, small local specialist, deterministic baseline, and hybrid workflow on identical cases.

## Acceptance criteria

- Reproducible toy experiments with known ground truth.
- Immutable baseline checkpoints and reversible candidate patches.
- Tensor allowlists and patch budgets enforced in code.
- Independent evaluation, held-out tests, and distribution-shift checks.
- No optimizer access to hidden evaluation data or ability to alter success criteria.
- No self-approval or self-promotion.
- Failed, null, and harmful attempts remain recorded.
- No general capability claims based on a handful of toy demonstrations.

## Non-goals and safeguards

- No unrestricted write access to production model weights.
- No autonomous online weight updates from unreviewed user interactions.
- No optimizer-controlled evaluation or acceptance decisions.
- No publication of private prompts, user data, secrets, or sensitive activations.
- No assumption that a weight-space repair transfers across model architectures or distributions.

## Relationship to other WeightLiftAI work

- [Issue #4](https://github.com/Luisv8181/weightliftai/issues/4): agent-assisted experiment orchestration.
- [Issue #6](https://github.com/Luisv8181/weightliftai/issues/6): Jacobian Lens and J-space representation analysis.
- [Issue #8](https://github.com/Luisv8181/weightliftai/issues/8): Weight-Space Programming and specialist optimizers.
- `docs/AGENT_EXPERIMENT_SYSTEM.md`: experiment lifecycle, bounded agent roles, and controls.
- `docs/JACOBIAN_LENS_RESEARCH.md`: activation-to-output influence analysis.

These tracks complement one another: mechanistic analysis helps generate hypotheses, weight-space programming proposes parameter changes, and the experiment harness determines whether the changes actually work.
