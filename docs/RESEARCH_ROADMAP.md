# Research Roadmap

## Phase 0 — Foundation
- [x] Establish project mission and scope
- [x] Define modular architecture
- [x] Define experiment and reproducibility protocol
- [x] Add Python package skeleton
- [x] Add hardware diagnostics
- [x] Add model inspection CLI (configuration-only by default; optional parameter loading)
- [x] Add initial unit tests and quick-start example
- [x] Run CI and verify on CPU and CUDA environments (verified on GitHub Actions Ubuntu CPU + local Windows 11 NVIDIA RTX 5070 CUDA 12.8)

## Phase 1 — Inspect
- [x] Report model configuration and parameter counts when weights are loaded
- [x] Enumerate selected module names and tensor shapes (first 500 entries)
- [x] Save a provenance manifest
- [ ] Add memory-aware dtype/device-map controls
- [ ] Add richer sharded-checkpoint metadata inspection

## Phase 2 — Observe
- [x] Capture selected module outputs with temporary forward hooks
- [x] Report descriptive activation statistics without saving raw tensors
- [x] List leaf modules and exact names through the CLI
- [x] Add side-by-side activation comparisons across prompts (`weightliftai compare`)
- [x] Add activation visualizations (SAE feature heatmap, word highlights, feature map, live trace replay: `feature_viz`, `feature_trace`)
- [ ] Document limits of neuron-level interpretation in tutorials

## Phase 3 — Adapt
- [ ] Add LoRA/PEFT baseline
- [ ] Add dataset validation and split checks
- [ ] Compare base model and adapter
- [ ] Add regression benchmark suite

## Phase 4 — Weight-space experiments
- [ ] Implement reversible tensor-edit interface
- [ ] Add task-vector arithmetic for compatible checkpoints
- [ ] Track source and candidate checksums
- [ ] Add strict shape/dtype compatibility validation
- [ ] Test rollback and artifact integrity

## Phase 5 — Assisted research
- [ ] Define machine-readable experiment proposals
- [ ] Let a supervisory model propose hypotheses, not silently execute them
- [ ] Require human approval for interventions
- [ ] Automatically run predeclared evaluation and regression tests
- [ ] Publish successful and unsuccessful experiments

## Out of scope for early releases
- Claims of autonomous self-improvement
- Automatic edits to users' source checkpoints
- Claims that one neuron represents one concept
- Clinical deployment or use with sensitive records

## Active 5-Part Research Program
- [x] **Experiment 001 — Bilingual Emotion SAE Map (`Gemma 3 1B` + `Qwen2.5-0.5B`):** Executed Layer-13 Gemma Scope 2 SAE mapping, Neuronpedia feature grounding, causal steering, 4-layer live token tracing, and LoRA drift check.
- [x] **Experiment 002 — Cultural Idioms of Distress & Counterfactual De-confounding:** Executed 3-way parallel split (English / Pan-Hispanic Neutral Spanish / Dominican Spanish idioms) across layers 7, 13, 17, 22 + 2×2 counterfactual feature clamping (`#1886`, `#4284`, `#293`). See [`docs/ROLLING_PAPER.md`](ROLLING_PAPER.md) §5.
- [ ] **Experiment 003 — Transcoder Causal Attribution Circuits:** Cross-layer feature attribution graphs using `gemma-scope-2-1b-it-transcoders` across English and Spanish.
- [ ] **Experiment 004 — Mechanistic Auditing of Clinical LoRA:** Evaluate SAE manifold drift (`FVE`), centroid separation, and feature turnover across LoRA ranks on therapeutic dialogue datasets.
- [ ] **Experiment 005 — Cross-Lingual Hub Intervention Transfer:** Compare middle-layer-only English LoRA (Layers 10–16) vs. all-layer LoRA vs. Layer-13 Contrastive Activation Steering on Spanish performance.


## Representation Cartography
- [x] **Phase 0 — Core Schemas, Validation Engine & Medical Terminology Pilot:**
  - Implemented immutable Pydantic schemas (CaptureSpec, RepresentationArtifact, ProjectionArtifact, MedicalConcept, ConceptSynonym, KnowledgeProbe, DatasetRef, ModelRef) in \src/weightliftai/cartography/schemas.py\.
  - Implemented automated cross-split leakage, matched pair balance, and self-referential cycle validation in \src/weightliftai/cartography/validators.py\.
  - Constructed the 50-concept NLM MeSH 2026 Medical Pilot dataset (\examples/medical_pilot/medical_pilot_50_concepts.json\): 150 balanced probes across 3 variants (canonical_clinical, plain_lay, cultural_idiom), 0 PHI, public domain.
- [x] **Phase 1 — Layer Atlas & Trajectory Engine:**
  - Implemented single-pass multi-depth residual stream capture (\extract_layer_representations\), paired bootstrap 95% CIs (\compute_layer_metrics\), and self-contained interactive HTML reporting (\generate_atlas_html\) in \src/weightliftai/cartography/layer_atlas.py\.
  - Added CLI subcommand \weightliftai atlas\ with automated depth sampling, token masking, and format reporting.
  - Executed Medical Pilot Layer Atlas on NVIDIA RTX 5070 across \google/gemma-3-1b-it\ (26 layers) and \google/gemma-4-E2B-it\ (35 layers).
  - **Empirical Validation of the Interlingua Hourglass Law on Clinical Knowledge:** Alignment across clinical, lay, and cultural idioms peaks precisely at the 50% structural midpoint (Gemma 3 L13: cos = 0.994 [0.994, 0.995]; Gemma 4 L17: cos = 0.981 [0.980, 0.982]) and collapses in output layers (Gemma 3 L25 drops to 0.900, Delta = 0.094 drop).
- [ ] **Phase 2 — Representation Map (2D/3D Dimensionality Reduction with Distortion Metrics):**
  - UMAP/PaCMAP/PCA projections with Trustworthiness & Continuity preservation metrics, original-space nearest-neighbor verification, and interactive point inspection.
- [ ] **Phase 3 — Sparse Feature Observatory & Grounding:**
  - Integration with SAE dictionaries, Top-K active feature attribution, Neuronpedia labeling, and sparsity statistics.
- [ ] **Phase 4 — Contrast Studio & Counterfactual Inspector:**
  - Matched-pair counterfactual diffs, dialect separation vectors, and semantic control verification.
- [ ] **Phase 5 — Circuit & Intervention Workbench:**
  - Reversible activation patching, feature clamping, and ablation impact analysis with before/after behavioral evaluation.
