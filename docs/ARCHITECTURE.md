# WeightLiftAI Architecture

## Design goal

Build a modular, local-first experiment harness that separates model loading, inspection, activation observation, intervention, evaluation, and artifact tracking.

## Core data flow

```text
Model + tokenizer + revision
          |
          v
    Model Loader
          |
          +------> Hardware report
          |
          v
    Model Inspector --------> Model manifest
          |
          v
  Baseline Evaluation ------> Baseline results
          |
          v
  Activation Explorer -----> Selected module summaries
          |
          v
  Experiment proposal
          |
          v
  Intervention adapter
  (LoRA / task vector /
   explicit tensor edit)
          |
          v
  Candidate model artifact
          |
          v
  Held-out evaluation + regression suite
          |
          v
  Comparison report + experiment record
```

## Module boundaries

- **hardware:** detect CPU, CUDA availability, device name, VRAM when accessible.
- **inspect_model:** report model config, parameters, layer/module names, tensor shapes and dtypes.
- **activations:** attach temporary forward hooks to exact named modules; summarize output tensors and always remove hooks after the forward pass.
- **manifest:** serialize provenance and environment metadata.
- **evaluation:** run controlled benchmark prompts and store outputs/scores.
- **interventions:** future module; all transformations must be explicit, isolated, and reversible.
- **registry:** future module for indexing runs and artifacts.

## Activation analysis behavior

The first activation explorer captures outputs from user-selected module names for a single forward pass. It reports shape, dtype, finite fraction, mean, standard deviation, min/max, mean absolute value, maximum absolute value, zero fraction, and a small list of largest absolute values with flat indices.

Raw activation tensors are not written to disk. Activation summaries may still disclose information about prompts, so users should only submit non-sensitive text. A module output statistic does not establish semantic meaning, causal importance, or a one-neuron/one-concept correspondence.

## Safety and integrity constraints

1. Load source models read-only where possible.
2. Never modify cached source files in place.
3. Save candidate weights/adapters to a new experiment directory.
4. Require explicit opt-in for weight edits.
5. Preserve a hash or immutable identifier for source artifacts.
6. Separate training/development data from held-out evaluation data.
7. Report target-task gains alongside general regression results.
8. Treat generated intervention proposals as untrusted hypotheses.
9. Remove temporary hooks even if the model forward pass raises an exception.

## Consumer GPU strategy

Use small models and memory-aware loading. Support CPU fallback. Do not assume that a model fits in VRAM based only on parameter count: dtype, quantization, context length, activations, and framework overhead matter.

## Research boundary

Inspection and standard PEFT workflows are established engineering techniques. Automated semantic interpretation of arbitrary weights and reliable direct weight editing remain active research problems. The architecture must make that distinction visible in its UI and reports.
