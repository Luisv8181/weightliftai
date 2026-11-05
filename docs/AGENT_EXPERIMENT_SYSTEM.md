# Agent-Assisted Experiment System

## Purpose

WeightLiftAI is moving toward a model behavior engineering workbench. Mechanistic interpretability is the foundation; controlled interventions and measured behavioral change are the product direction.

The experiment system is designed around:

**Baseline → Hypothesis → Intervention → Evaluation → Comparison → Independent review → Reproducible artifact**

Agents can help plan, implement, execute, analyze, and report experiments. The experiment harness—not an agent's narrative—must determine whether preregistered criteria were met.

## Current implementation

The first implementation is a local-first append-only experiment registry in `src/weightliftai/experiment_registry.py`.

It currently provides:
- A manifest builder requiring a model ID, immutable model revision, dataset ID, split, primary metric, and success criterion.
- Safe experiment IDs and create-only run directories.
- A lifecycle state machine.
- Append-only JSONL event history.
- Provenance hashes for manifests and result sources.
- Human approval references before intervention execution and model promotion.
- Protection against silently overwriting baseline or intervention result records.

It deliberately does **not** execute model code, load checkpoints, launch agents, edit weights, or claim that an intervention works. Those require later isolated execution and evaluation components.

## Experiment record

Each experiment should capture:

- Hypothesis and preregistered success criterion.
- Model ID, immutable revision, tokenizer revision, architecture, dtype, quantization, and runtime.
- Dataset ID, version/hash, provenance, license/consent, split, and leakage controls.
- Prompt templates, decoding parameters, seeds, software environment, hardware, and code commit.
- Intervention type, target component/layer/feature/tensor, exact parameters, and patch hash.
- Baseline, intervention, sham/control, and regression metrics.
- Raw artifact hashes and paths; private raw data should remain local unless explicitly approved.
- Agent identity/model, role, prompt, tools used, code diff, commands, and resource consumption.
- Reviewer decision, limitations, and rollback instructions.

Do not put API keys, secrets, private prompts, identifiable data, or large checkpoint files in manifests or public Git history.

## Lifecycle

The registry enforces these transitions:

`draft → preregistered → baseline_complete → intervention_ready → intervention_complete → evaluated → independently_reviewed → accepted | rejected | inconclusive`

- A run starts in `draft`.
- The protocol and metrics are frozen at `preregistered`.
- Baseline artifacts are captured before intervention.
- A human approval reference is required to enter `intervention_ready`.
- Evaluation must include the target metric and preregistered regression gates.
- An independent reviewer must assess the evidence.
- A human promotion approval reference is required before `accepted`.
- Rejected and inconclusive experiments remain in the registry. Do not delete null or failed results.
- Replications should be new experiments linked through `parent_experiment_id`.

## Agent roles and permissions

| Role | Allowed | Not allowed |
|---|---|---|
| Research planner | Propose hypotheses, controls, metrics, and experiment manifests | Edit results, change frozen metrics, approve own work |
| Implementation agent | Create code in an isolated branch/sandbox | Access secrets, overwrite baseline, modify protected test data |
| Execution agent | Run approved manifests within resource budgets | Run unapproved weight edits or arbitrary unrestricted shell code |
| Analysis agent | Calculate metrics from recorded raw outputs | Drop failed runs or redefine success criteria |
| Skeptic/reviewer | Look for confounds, leakage, cherry-picking, and regressions | Alter raw results |
| Report agent | Generate reports from registry records | Claim success without passing gates |

Human approval is required before persistent parameter edits, expensive compute, dataset publication, and model artifact promotion.

## Intervention ladder

Compare interventions on the same task, split, and evaluation protocol:

1. Prompt/context baseline.
2. Retrieval or external knowledge baseline.
3. Inference-time activation steering or patching.
4. SAE feature interventions, only when the feature dictionary and feature semantics are adequately validated.
5. LoRA/PEFT.
6. Targeted weight-space editing.
7. Full fine-tuning as a comparator or fallback.

Weight edits are not presumed to be the best solution. Compare effect size, regressions, generalization, compute, latency, reversibility, and operational complexity.

## Scientific requirements

- Freeze test sets and primary metrics before seeing intervention results.
- Use held-out prompts, topics, speakers, and lexical items where relevant.
- Use negative/sham controls and unrelated capability checks.
- Use multiple seeds or replications when feasible.
- Report uncertainty and sample sizes.
- Use human evaluation for pragmatic naturalness, cultural appropriateness, and other qualities not captured by automated metrics.
- Separate decodability, correlation, causal contribution, and mechanistic explanation.
- Record every attempt, including failed and null results.
- Never infer a dedicated concept neuron from a probe, cosine similarity, or 2D visualization alone.

## Experiment memory

A future analysis layer can aggregate prior experiments to help propose better next tests. It may learn empirical patterns such as:
- Intervention families that repeatedly help or harm a task.
- Layers or components that are promising under a specific architecture and dataset.
- Common regression signatures.
- Which controls exposed confounds.
- Which findings replicated across models, seeds, and datasets.

Experiment memory must retain evidence links and uncertainty. It should distinguish replicated findings from one-off observations and must not convert correlation into causal claims.

## Beyond weight space

Weight space is one of several coupled control surfaces:

1. Data and training history.
2. Architecture and learned parameters.
3. Representations and computation (activations, residual stream, attention, MLPs, features, circuits).
4. Inference-time controls (steering, decoding, sampling, test-time compute).
5. Context and knowledge (prompts, retrieval, memory, tools, external state).
6. Agent/system orchestration (planning, tool permissions, policies, feedback).
7. Application and human environment (workflows, users, evaluation, real-world outcomes).

The platform should identify the least invasive intervention that solves the measured problem. A retrieval or context fix may be more reliable than changing weights.

## First pilot

Use the Dominican Spanish and culturally grounded translation research in GitHub Issue #3. Start with a small inspectable open-weight text model and a speaker-diverse, consented, reviewed evaluation set. Treat TTS as a separate model family and feasibility study; text-model findings do not automatically transfer to speech synthesis.

Suggested sequence:
1. Freeze model revision, dataset, prompts, and metrics.
2. Run and archive baseline.
3. Have agents propose candidate mechanisms and controls.
4. Human-review the experiment manifest.
5. Compare prompt/context, activation steering, adapter, and weight-edit methods only where technically feasible.
6. Evaluate on held-out speakers/prompts and general Spanish controls.
7. Publish target gains, regressions, null results, uncertainty, and limitations.

## Local usage

The registry API can be imported directly:

```python
from weightliftai.experiment_registry import (
    build_experiment_manifest,
    create_experiment,
    transition_experiment,
)

manifest = build_experiment_manifest(
    hypothesis="The intervention improves a preregistered behavior.",
    model={"model_id": "org/model", "revision": "immutable-commit"},
    evaluation={
        "dataset_id": "pilot-v1",
        "split": "test",
        "primary_metric": "task_metric",
        "success_criterion": "predefined threshold",
    },
)

run_dir = create_experiment("local_experiments", manifest)
transition_experiment(
    run_dir, "preregistered",
    actor="researcher",
    rationale="Protocol and held-out evaluation were frozen.",
)
```

The complete lifecycle is intentionally strict. See the module docstrings and tests for current API behavior.

## Roadmap

- [x] Initial manifest builder and validator.
- [x] Safe run directory creation and manifest hash.
- [x] Append-only JSONL event log.
- [x] Lifecycle state machine and human approval references.
- [x] Result provenance hashes and overwrite protection.
- [ ] CLI commands for creating, validating, and inspecting experiments.
- [ ] Deterministic baseline/evaluation runner.
- [ ] Dataset and model provenance capture helpers.
- [ ] Baseline/intervention/control comparison and uncertainty reports.
- [ ] Isolated agent execution with explicit permissions and resource budgets.
- [ ] Activation/SAE/LoRA/weight-edit intervention adapters.
- [ ] Independent reviewer reports and promotion gates.
- [ ] Evidence-linked experiment memory and recommendation engine.
