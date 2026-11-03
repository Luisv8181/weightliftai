# Research Roadmap

## Phase 0 — Foundation
- [x] Establish project mission and scope
- [x] Define modular architecture
- [x] Define experiment and reproducibility protocol
- [ ] Add Python package skeleton
- [ ] Add hardware diagnostics
- [ ] Add model inspection CLI
- [ ] Add tests and quick-start example

## Phase 1 — Inspect
- [ ] Load a small supported open-weight model
- [ ] Report model metadata and parameter counts
- [ ] Enumerate selected module names and tensor shapes
- [ ] Save a provenance manifest
- [ ] Add CPU and CUDA smoke tests

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
