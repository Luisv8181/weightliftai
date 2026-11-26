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

## Milestones & Capabilities

WeightLiftAI is designed as an open-weight model interpretability, evaluation, and intervention laboratory.

### Milestone 0 — Reproducible local baseline
- [x] Detect available hardware and report CUDA/CPU diagnostics (`weightliftai hardware`).
- [x] Load supported Hugging Face causal language models (`google/gemma-3-1b-it`, `Qwen/Qwen2.5-0.5B-Instruct`).
- [x] Print model configuration, layer count, parameter count, and tensor shapes (`weightliftai inspect`).
- [x] Save immutable baseline manifests with environment provenance (`weightliftai.manifest`).
- [x] Run evaluation sets and save artifacts without modifying base model weights.

### Milestone 1 — Multi-depth activation & feature inspection
- [x] Capture selected activations with temporary forward hooks and guarantee cleanup on exceptions.
- [x] Summarize descriptive activation statistics without storing raw multi-gigabyte tensors to disk.
- [x] List leaf modules and exact sub-layer names (`weightliftai modules`).
- [x] Multi-layer Sparse Autoencoder (SAE) encoding and tracing via SAELens across depths (`7, 13, 17, 22`).
- [x] Publication-grade interactive visualization suite:
  - 16,384-feature concept heatmaps (`concept_heatmap_html`)
  - Word-level activation glow highlights (`token_highlight_html`)
  - 2D constellation feature maps (`feature_map_html`)
  - Animated token-by-token generation replays across 4 layers (`trace_replay_html`)
- [x] Safe, explicit, reversible interventions:
  - Causal SAE feature steering with customizable strength dials
  - Error-preserving residual stream feature clamping (`h + dec(f_zero) - dec(f)`)

### Milestone 2 — Adaptation comparisons & audit harness
- [x] Add LoRA/PEFT baseline adapter training with non-destructive in-memory attachments.
- [x] Pre-registered mechanistic audit protocol: evaluate SAE manifold drift (`FVE`), centroid similarity, and held-out cross-lingual transfer before vs. after adaptation.
- [ ] Scale adaptation from pilot format checks to full multi-turn clinical dialogue corpora (`counsel-chat`, `esconv`).

### Milestone 3 — Weight-space & circuit research
- [x] Causal shortcut de-confounding via 2×2 counterfactual splits and feature zero-clamping.
- [x] Rigorous publication of negative and falsified results in the living paper claims register.
- [ ] Cross-layer causal attribution graphs with Gemma Scope 2 MLP transcoders.
- [ ] Task-vector arithmetic and targeted weight edits across compatible model checkpoints.

## Representation Cartography

**Maturity Status:** `tested_implementation` (`layer_atlas.py`, `representation_map.py`, `schemas.py`, `validators.py`).
A local-first research module for mapping model representations across layers, languages, dialects, and affective concepts. It connects layer-wise activation capture, original-space similarity metrics, distortion-checked 2D/3D projections (`T(k)`, `C(k)`, `k`-NN Jaccard), and matched counterfactuals. Visual projections are exploratory aids, not literal pictures of a model's conceptual space. See [**`docs/REPRESENTATION_CARTOGRAPHY.md`**](docs/REPRESENTATION_CARTOGRAPHY.md) and [**`docs/RESEARCH_VALIDATION_PROTOCOL.md`**](docs/RESEARCH_VALIDATION_PROTOCOL.md).

## Jacobian Lens & J-Space Research

**Maturity Status:** `tested_implementation` (cross-position VJP `d h[L, t'] / d h[l, t]` for `t' >= t`, multi-context/position aggregation, and 4-dimension evidence separation in `src/weightliftai/cartography/jacobian_lens.py`) / `proposed_research_specification` (overcomplete sparse nonnegative J-space dictionary decomposition).
Extends representation cartography from similarity/probe analysis toward output-influence analysis while keeping J-space distinct from parameter/weight space. Separates observational cosine alignment, VJP sensitivity, cross-context/position stability, and controlled activation intervention effects against random and orthogonal sham controls. See [**`docs/JACOBIAN_LENS_RESEARCH.md`**](docs/JACOBIAN_LENS_RESEARCH.md) and [Issue #6](https://github.com/Luisv8181/weightliftai/issues/6).

## Weight-Space Programming & Continuous Model Improvement

**Maturity Status:** `tested_implementation` (evaluator-authoritative blind benchmark `WSP-001A`, LLM/gradient/deterministic proposer comparison `WSP-001B`, and sandboxed patch budgets in `src/weightliftai/weight_programming/`).
Explores whether language models and optimization algorithms can inspect bounded neural-network diagnostics and propose reversible repairs under an independent, process-isolated evaluator with Ed25519 signed receipts. See [**`docs/WEIGHT_SPACE_PROGRAMMING.md`**](docs/WEIGHT_SPACE_PROGRAMMING.md), [**`docs/BLIND_BENCHMARK_THREAT_MODEL.md`**](docs/BLIND_BENCHMARK_THREAT_MODEL.md), and [Issue #8](https://github.com/Luisv8181/weightliftai/issues/8).

## Agent Handoff & Medical Knowledge Research

For coding agents and collaborators, start with [**`docs/AGENT_HANDOFF.md`**](docs/AGENT_HANDOFF.md). It summarizes the current research direction, prior work and tools to learn from, operating instructions, immediate priorities, and a proposed medical knowledge research track. The medical track separates terminology/knowledge grounding, multilingual clinical language, model representations, and safety evaluation; it is research-only and explicitly excludes patient data and clinical deployment.

## Experiments

> 📄 **Rolling Research Paper & Active Protocol:** For the active experimental protocol (DSB-1 and phased plan starting with Exp 015), see [**`docs/RESEARCH_PROTOCOL.md`**](docs/RESEARCH_PROTOCOL.md). For the living manuscript, confidence intervals, and claims register, see [**`docs/ROLLING_PAPER.md`**](docs/ROLLING_PAPER.md). Automated run logs are tracked in [**`docs/AUDIT_LEDGER.jsonl`**](docs/AUDIT_LEDGER.jsonl).

| ID | Question | Status |
|---|---|---|
| [001](experiments/001-gemma-emotion-sae-map/) | Does Gemma 3 1B keep grief vs. sadness and fear vs. anxiety distinct, in English and Spanish? Uses Gemma Scope 2 sparse-autoencoder features, interactive feature views, a live word-by-word trace across four layers, steering, and a LoRA before/after comparison. | **Executed** on RTX 5070 (`FVE=0.743`, EN CV `92.5%`, ES CV `65.0% -> 72.5%` after LoRA, cross-lingual transfer `95.0%` EN→ES / `100.0%` ES→EN) |
| [002](experiments/002-dialect-and-counterfactuals/) | Cultural idioms of distress (Dominican Spanish vs. Pan-Hispanic Spanish vs. English) and counterfactual lexical shortcut de-confounding (`still` / `death` / `family` feature clamping). | **Executed** on RTX 5070 (Part A: 3-way parallel split across layers 7/13/17/22; Part B: 2×2 counterfactual proves top grief features are keyword shortcuts; probe trap FPR 1.00) |
| [003](experiments/003-tigre-rank1-weight-patch/) | Ungated rank-1 weight patching on `Qwen/Qwen2.5-0.5B-Instruct` and `google/gemma-4-E2B-it` for Dominican Spanish `"tigre"`/`"tíguere"` slang. | **Executed** on RTX 5070 (negative/partial result: `0.633` slang win rate, `0.000` homograph win rate, `+81.1%` NLL drift) |
| [004](experiments/004-cartography-and-gated-patch/) | 4-model layer-by-layer cartography + self-gated rank-1 weight patching on 100% homograph `"tigre"` corpus across `Qwen2.5-0.5B`, `Gemma 4 E2B`, `NVIDIA AceInstruct-1.5B`, and `HAL-X-AI/THX-02`. | **Partially replicated / exploratory (`n=8` test per class)** on RTX 5070 (`1.000` test probe accuracy at L16 on `Gemma 4` and `AceInstruct-1.5B`, `0.00%` general Spanish NLL drift on 8-item check) |
| [005](experiments/005-wsp-001b-llm-ablation/) | `WSP-001B` blind repair ablation comparing deterministic, gradient, and LLM proposers under evaluator-authoritative isolation. | **Executed** (harness & CI verified; see [WSP-001B report](docs/WSP_001B_ABLATION_AND_COMPARISON_REPORT.md)) |
| [006](experiments/006-jacobian-affective-cartography/) | Cross-position VJP Jacobian affective cartography on `Qwen2.5-0.5B-Instruct` vs. random/orthogonal shams. | **Inconclusive (n too small)** (pilot; see [Jacobian report](docs/JACOBIAN_AFFECTIVE_CARTOGRAPHY_REPORT.md)) |
| [007](experiments/007-astra-weight-surgery/) | Null-space key-gated Layer 14 MLP down-projection weight edit pilot on `Qwen2.5-0.5B-Instruct`. | **Inconclusive (n too small)** (`n=2` target idioms, `n=3` controls; see [Exp 007 report](docs/ASTRA_WEIGHT_SURGERY_REPORT.md)) |
| [008](experiments/008-joint-subspace-surgery/) | Rank-5 Moore-Penrose joint null-space projection across 5 Dominican idioms on `Qwen2.5-0.5B-Instruct`. | **Inconclusive (n too small)** (`n=5` targets, `n=8` controls; see [Exp 008 report](docs/EXP_008_JOINT_SUBSPACE_SURGERY_REPORT.md)) |
| [009](experiments/009-dialogic-coherence/) | Multi-turn generation inspection under the Exp 008 patch (research-only, not clinical). | **Inconclusive (n too small)** (qualitative example, `n=1` conversation / 3 prompts; see [Exp 009 report](docs/EXP_009_DIALOGIC_COHERENCE_REPORT.md)) |
| [010](experiments/010-wsp-paper-replication/) | MIT CSAIL WSP paper replication pilot: Part 1 (Weight Reading), Part 2 (Weight Writing), Part 3 (Loop Surgery). | **Part 1 & Part 2: Non-blind / invalid** (`n=4` / same agent wrote env & gold; Qwen/Gemma at 25% chance); **Part 3: Inconclusive (n too small)** (`n=1` qualitative example). See [CLAIM_EVIDENCE_MATRIX.md](docs/replication/CLAIM_EVIDENCE_MATRIX.md). |
| [011](experiments/011-math500-weight-surgery/) | Stratified `n=35` MATH-500 sweep on `Qwen/Qwen2.5-0.5B-Instruct` under Layer 14 rank-1 edit. | **Not replicated** (paper reported `7.9% -> 48.1%` on `Qwen3.5-0.8B`; this run showed no degradation `11.43% -> 11.43%` on `Qwen2.5-0.5B-Instruct`, `0/35` loops). See [MATH-500 report](docs/replication/MATH_500_REPLICATION_REPORT.md). |

## Research principles

- **Baseline first:** Never claim improvement without a fixed baseline and held-out evaluation.
- **Reversible changes:** Never overwrite original model weights. Save interventions as separate artifacts.
- **Regression testing:** Measure unrelated capabilities, not only the target task.
- **Reproducibility:** Record model revision, tokenizer, dataset version, seeds, software versions, hardware, and intervention parameters.
- **No magical neuron claims:** Individual neurons or heads are not assumed to correspond one-to-one with human concepts or tasks.
- **Privacy by design:** Keep evaluation data local by default. Do not use confidential or identifiable data in experiments without appropriate authorization and protections.
- **Human oversight:** Proposed modifications remain experimental until independently evaluated.

## Suggested stack & installation

- Python 3.11+
- Core: `pip install -e ".[core]"`
- Cartography & Jacobian Lens: `pip install -e ".[cartography]"`
- Development & test suite: `pip install -e ".[dev]"`

## Disclaimer

WeightLiftAI is an experimental research and developer tool. It does not guarantee improved intelligence, correctness, safety, or efficiency. Modifying model parameters can degrade behavior, introduce unexpected outputs, or invalidate assumptions made by the original model developers.
