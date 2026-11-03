# WeightLiftAI

**Local-first tools for inspecting, adapting, and evaluating open-weight neural networks.**

WeightLiftAI is an open-source research and engineering project exploring practical methods for understanding and improving locally runnable AI models. It aims to make model internals more accessible to developers working with consumer hardware.

> **Research posture:** WeightLiftAI distinguishes established techniques (inspection, activation analysis, LoRA, controlled fine-tuning, and benchmarking) from experimental direct weight-space interventions. Every claimed improvement must be reproducible and measured against an unchanged baseline.

## Why WeightLiftAI?

Most model customization workflows focus on prompts, retrieval, or fine-tuning. WeightLiftAI explores a broader engineering workflow:

1. Inspect a model and understand its structure.
2. Define a measurable behavioral weakness.
3. Build a reproducible baseline evaluation.
4. Investigate relevant weights and activations.
5. Apply a controlled intervention.
6. Evaluate improvements and regressions.
7. Save, compare, and roll back model variants.

The goal is not to make every model larger. It is to help developers understand what a model can do, where it fails, and which interventions are worth testing.

## Project pillars

| Pillar | Scope |
|---|---|
| **Model Inspector** | Load model metadata, configuration, layers, tensor shapes, and parameter counts. |
| **Activation Explorer** | Capture and compare selected module outputs and activation statistics on controlled inputs. |
| **Adaptation Lab** | Run parameter-efficient fine-tuning such as LoRA as a practical baseline. |
| **Weight Experimenter** | Apply explicit, reversible tensor interventions in isolated experiments. |
| **Benchmark Lab** | Compare baseline and modified models for task performance, regressions, and resource use. |
| **Experiment Registry** | Record model identifiers, revisions, configs, seeds, interventions, metrics, and artifacts. |

## Initial scope

The first release should be a **local model inspection and experiment harness**, not an autonomous self-modifying agent.

### Milestone 0 — Reproducible local baseline

- Detect available hardware and report CUDA/CPU availability.
- Load a supported Hugging Face causal language model.
- Print model configuration, layer count, parameter count, and tensor shapes.
- Save a baseline manifest.
- Run a small, user-provided evaluation set and save outputs.
- Keep model weights unchanged.

### Milestone 1 — Activation and tensor inspection

- [x] Capture selected activations with forward hooks.
- [x] Summarize activation statistics without saving raw activation tensors.
- [x] List leaf modules and inspect exact module names.
- [ ] Add richer activation comparison and visualization workflows.
- [ ] Add safe, explicit, reversible interventions behind an experimental flag.

### Milestone 2 — Adaptation comparisons

- Add LoRA/PEFT as a strong practical baseline.
- Compare full-precision and adapter-based variants where hardware permits.
- Track quality, latency, memory use, and regression tests.

### Milestone 3 — Weight-space research

- Explore task vectors and model merging where architectures and tensor shapes are compatible.
- Test targeted weight edits with strict safeguards.
- Investigate whether a supervisory model can propose interventions that humans review and benchmark.
- Publish negative results as well as positive results.

## Experiments

| ID | Question | Status |
|---|---|---|
| [001](experiments/001-gemma-emotion-sae-map/) | Does Gemma 3 1B keep grief vs. sadness and fear vs. anxiety distinct, in English and Spanish? Uses Gemma Scope 2 sparse-autoencoder features, steering, and a LoRA before/after comparison. | Set up (Colab notebook); not yet run |

## Research principles

- **Baseline first:** Never claim improvement without a fixed baseline and held-out evaluation.
- **Reversible changes:** Never overwrite original model weights. Save interventions as separate artifacts.
- **Regression testing:** Measure unrelated capabilities, not only the target task.
- **Reproducibility:** Record model revision, tokenizer, dataset version, seeds, software versions, hardware, and intervention parameters.
- **No magical neuron claims:** Individual neurons or heads are not assumed to correspond one-to-one with human concepts or tasks.
- **Privacy by design:** Keep evaluation data local by default. Do not use confidential or identifiable data in experiments without appropriate authorization and protections.
- **Human oversight:** Proposed modifications remain experimental until independently evaluated.

## Hardware and model support

The project prioritizes consumer GPUs and small open-weight models. The initial target is a workflow that can inspect and evaluate models that fit available system memory, with optional CUDA acceleration.

Initial model-family candidates:
- Gemma
- Qwen
- Llama
- Mistral

Actual support depends on model architecture, library compatibility, license, and available memory. Weight-space operations are not universally transferable between architectures.

## Suggested stack

- Python 3.11+
- PyTorch
- Hugging Face Transformers
- Accelerate
- PEFT
- Safetensors
- pytest
- Optional: MLflow for experiment tracking

## Repository structure (target)

```text
weightliftai/
├── README.md
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── ARCHITECTURE.md
│   ├── RESEARCH_ROADMAP.md
│   └── EXPERIMENT_PROTOCOL.md
├── src/weightliftai/
│   ├── hardware.py
│   ├── inspect_model.py
│   ├── activations.py
│   ├── manifest.py
│   └── cli.py
├── experiments/
├── tests/
└── examples/
```

## Status

**Research foundation — early development.**

The repository currently establishes the project's scope and research principles. Runnable capabilities will be added incrementally and documented with tested commands.

## Contributing

Issues and pull requests are welcome. Please include the problem and hypothesis, exact model revision, reproduction steps, environment, baseline/comparison metrics, and known limitations or regressions.

## Disclaimer

WeightLiftAI is an experimental research and developer tool. It does not guarantee improved intelligence, correctness, safety, or efficiency. Modifying model parameters can degrade behavior, introduce unexpected outputs, or invalidate assumptions made by the original model developers.
