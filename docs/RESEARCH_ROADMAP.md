# Research Roadmap

## Phase 0 — Foundation
- [x] Establish project mission and scope
- [x] Define modular architecture
- [x] Define experiment and reproducibility protocol
- [x] Add Python package skeleton
- [x] Add hardware diagnostics
- [x] Add model inspection CLI (configuration-only by default; optional parameter loading)
- [x] Add initial unit tests and quick-start example
- [ ] Run CI and verify on CPU and CUDA environments

## Phase 1 — Inspect
- [x] Report model configuration and parameter counts when weights are loaded
- [x] Enumerate selected module names and tensor shapes (first 500 entries)
- [x] Save a provenance manifest
- [ ] Add memory-aware dtype/device-map controls
- [ ] Add richer sharded-checkpoint metadata inspection

## Phase 2 — Observe
- [ ] Capture layer activations on a small controlled input
- [ ] Compare activation statistics across examples
- [ ] Add simple visual summaries
- [ ] Document limits of neuron-level interpretation

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
