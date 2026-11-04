# Representation Cartography — Product & Research Plan

**Status:** Proposed product direction and implementation roadmap. The visual explorer is not yet implemented.

## Product thesis

Representation Cartography makes model internals explorable as evidence-bearing maps. Researchers follow controlled inputs through a model, inspect how representations change by layer, compare matched examples across languages and dialects, examine candidate SAE features, and test causal hypotheses. The interface must never confuse a visualization with an explanation.

**Core promise:** Map representations. Trace transformations. Test hypotheses. Preserve evidence.

## Research questions

1. At which layers are matched meanings across languages more or less geometrically aligned?
2. Do matched semantic items expressed in Dominican Spanish, broader Spanish varieties, and English produce systematically different representations?
3. Which affective distinctions are decodable, and do they survive keyword-free and counterfactual tests?
4. Which SAE features respond robustly to a contrast rather than lexical shortcuts?
5. Does intervening on a candidate feature or component change target behavior while preserving unrelated capabilities?
6. Which patterns replicate across model families, sizes, revisions, and seeds?

## Researcher workflow

1. Select a local model and immutable revision.
2. Select/import a versioned, non-sensitive, matched probe dataset.
3. Define contrasts: language, dialect, concept, context, and counterfactual condition.
4. Choose capture sites: residual stream, attention output, MLP output, or SAE features.
5. Extract representations with explicit token masking and pooling.
6. Explore layer-wise maps and original-space quantitative metrics.
7. Select a point to inspect its item ID, text (if safe), metadata, neighbors, feature responses, and provenance.
8. Register a hypothesis before testing.
9. Run held-out and causal tests in an isolated model copy.
10. Export a report with methods, uncertainty, limitations, and audit references.

## Product surfaces

### 1. Layer Atlas
- Synchronized layer/depth selector and timeline.
- Per-layer representation statistics and transfer probes.
- Compare early, middle, and late layers without assuming fixed semantic roles.
- Support residual stream, attention output, and MLP output capture sites.

### 2. Representation Map
- PCA as the first deterministic 2D/3D projection; optional UMAP/t-SNE as exploratory views.
- Color/shape by language, dialect, concept, source, and counterfactual condition.
- Select points to reveal original item ID, source text (if safe), model revision, layer, and capture settings.
- Nearest-neighbor search in the original high-dimensional space.
- Show projection parameters and stability/trustworthiness diagnostics.
- Never use screen-space distances as scientific evidence.

### 3. Feature Observatory
- SAE feature activation heatmaps and token-level traces.
- Show feature ID, dictionary release, layer, activation distributions, top positive and negative examples.
- Compare features across matched languages, dialects, and lexical counterfactuals.
- Separate automated labels from human-verified interpretations.
- Report hit rates, false positives/negatives, effect sizes, and held-out results.

### 4. Contrast Studio
- Side-by-side prompts and matched item groups.
- Compare cosine similarity, L2 distance, CKA/subspace similarity, probe performance, and layer-wise trajectories.
- Record tokenization, pooling, BOS/EOS/padding masks, normalization, dtype, and seeds.
- Support paired item bootstrap and confidence intervals.

### 5. Circuit & Intervention Workbench (later phase)
- Attribution graphs and cross-layer component pathways.
- Activation patching, ablation, SAE feature clamping/steering, and supported transcoder workflows.
- Run interventions only on isolated model copies; make them reversible.
- Require baselines, held-out target evaluation, unrelated-task regression checks, and human approval.
- Clearly distinguish correlational feature maps from causal circuit evidence.

### 6. Experiment Passport
- Every map and comparison links to an immutable run record.
- Store model/revision, tokenizer, dataset hash, code commit, hardware, library versions, capture modules, pooling/masking, projection parameters, seeds, metrics, and artifact hashes.
- Raw text and raw activation tensors are opt-in; default to local processing and minimal artifacts.
- Export figures and machine-readable JSON/CSV summaries.

## Scientific and UX guardrails

- **Projection is not geometry:** 2D/3D maps are lossy projections. Compute quantitative comparisons in original vector space or with validated subspace metrics.
- **Decodability is not use:** A probe can reveal accessible information; it does not prove the model relies on it.
- **Correlation is not causation:** Feature activation alone does not establish semantic meaning or causal function.
- **Feature is not neuron:** SAE feature IDs must never be presented as neuron indices.
- **Matched items matter:** Compare matched semantic items; report item-level uncertainty and avoid treating paraphrases as independent samples.
- **Dialect is not a single axis:** Record region, register, context, provenance, and community review. Generated dialect text is not authoritative.
- **Human review:** Engage fluent speakers and cultural/domain reviewers before interpreting dialectal or affective patterns.
- **Clinical boundary:** Use public, synthetic, de-identified, or otherwise authorized research data only. No diagnosis, clinical decision support, or patient-conversation deployment from this exploratory tool.
- **Reproducibility:** Pin revisions, data hashes, seeds, software, and capture configuration. Keep negative results and withdrawn claims visible.

## Technical architecture

### Initial data objects
- `ModelRef`: model ID, revision, tokenizer revision, architecture metadata, dtype.
- `DatasetRef`: dataset ID/version/hash, item IDs, language/dialect/concept labels, provenance and consent status.
- `CaptureSpec`: module path, layer index, activation site, token mask, pooling, normalization.
- `RepresentationArtifact`: item IDs, vectors or secure local artifact pointer, capture spec, model/dataset hashes.
- `ProjectionArtifact`: algorithm, parameters, random seed, fitted transform, input artifact hash, coordinates.
- `FeatureArtifact`: SAE release/dictionary, layer, feature IDs, activations, token alignment.
- `ComparisonRun`: metric, matched pairs, estimates, intervals, split strategy.
- `InterventionRun`: target components/features, intervention type/strength, source checkpoint, candidate artifact, evaluation and regression results.

### Pipeline
```text
Model + revision ─┐
Dataset + hashes ─┼─> CaptureSpec ─> Activation extraction
Hardware/runtime ─┘                         |
                                             v
                                 Representation artifacts
                                             |
                       ┌─────────────────────┼─────────────────────┐
                       v                     v                     v
                 Layer Atlas          Feature Observatory     Contrast Studio
                       |                     |                     |
                       └─────────────────────┼─────────────────────┘
                                             v
                                    Projection / metrics
                                             |
                                             v
                                      Experiment Passport
                                             |
                                 hypothesis + human approval
                                             |
                                             v
                                Intervention Workbench
                                             |
                              held-out + regression evaluation
                                             |
                                             v
                                  audited report / registry
```

### Implementation approach
- Reuse `activations.py`, `feature_viz.py`, `feature_trace.py`, existing experiments, and the audit ledger; do not duplicate model hooks.
- Generalize tested capture and dataset-alignment interfaces before building a frontend.
- Start with local static HTML or a small local web UI after the artifact schema stabilizes.
- PCA is the baseline; UMAP/t-SNE are optional exploratory projections.
- Preserve original high-dimensional vectors (or local references) for metrics and nearest-neighbor queries.
- Prioritize small local models and memory-aware batches; no hosted inference requirement.

## Phased delivery

### Phase 0 — Research contract and schemas
- [ ] Finalize data model and artifact versioning.
- [ ] Add dataset manifest with matched-item validation and dialect/provenance fields.
- [ ] Define capture specification and masking/pooling conventions.
- [ ] Write a guide to projection and metric interpretation.

### Phase 1 — Layer Atlas (first usable release)
- [ ] Generalize activation capture to save compact, reproducible representation artifacts.
- [ ] Capture selected layers/sites in one controlled forward pass.
- [ ] Add layer-wise cosine, L2, linear-probe, and paired-bootstrap summaries.
- [ ] Create a static layer-trajectory report from existing Gemma 3, Qwen, and Gemma 4 runs.
- [ ] Include baseline and dataset limitations on every report.

### Phase 2 — Representation Map
- [ ] Add PCA with deterministic seeds and explained-variance metadata.
- [ ] Add optional UMAP/t-SNE with explicit exploratory labeling.
- [ ] Add point selection, matched-item links, nearest neighbors, and filters.
- [ ] Add projection stability and trustworthiness diagnostics.

### Phase 3 — Feature Observatory
- [ ] Integrate SAE activations with maps and token-level traces.
- [ ] Compare feature distributions across matched language/dialect/counterfactual groups.
- [ ] Add human feature-grounding review and false-positive/negative examples.
- [ ] Reconcile Experiment 002b feature #2323 with the claims register and replace zero-denominator ratios with an appropriate statement.

### Phase 4 — Cross-model and cultural validation
- [ ] Run repeated stratified splits and multiple seeds.
- [ ] Add independent held-out probe sets and item-level bootstrap intervals.
- [ ] Obtain native-speaker review for Dominican Spanish and additional dialect sets.
- [ ] Replicate with another model family and report failed replications.
- [ ] Compare original-space metrics, not only 2D projections.

### Phase 5 — Circuit Explorer and causal workbench
- [ ] Add supported transcoder/attribution graph integrations.
- [ ] Implement activation patching and feature ablation on isolated model copies.
- [ ] Require pre-registered hypotheses and human approval.
- [ ] Evaluate target behavior and unrelated capabilities on held-out data.
- [ ] Preserve intervention provenance and rollback artifacts.

## First-release acceptance criteria

- One command captures at least five selected layers for a versioned matched dataset.
- The run is reproducible from its manifest and code commit.
- Reports include layer-wise metrics and uncertainty, not only a projection.
- Every plotted point maps to an item ID and capture configuration.
- Projection coordinates are never substituted for original-space metrics.
- No raw sensitive text is stored by default.
- Tests cover padding/BOS masking, variable sequence lengths, deterministic extraction, and hook cleanup after exceptions.

## Prior work and tools to learn from

- **Tak et al. (2025), emotion mechanisms:** *Mechanistic Interpretability of Emotion Inference in Large Language Models*, Findings of ACL. Directly relevant to emotion representations, appraisal concepts, and causal intervention.
- **Chang, Tu & Bergen (2022), multilingual geometry:** *The Geometry of Multilingual Language Model Representations*, EMNLP. A foundation for shared vs. language-sensitive subspaces and geometric analysis.
- **Havaldar et al. (2023), cultural emotion:** *Multilingual Language Models are not Multicultural: A Case Study in Emotion*. Motivates separating language coverage from cultural understanding.
- **Zhang et al. (2025), multilingual circuits:** *The Same but Different: Structural Similarities and Differences in Multilingual Language Modeling*, ICLR. Studies shared and language-specific circuitry.
- **Ben Tamo et al. (2026), layer localization:** *LinguaMap: Which Layers of LLMs Speak Your Language and How to Tune Them?*, ICLR. Directly relevant to layer-wise language control and selective tuning.
- **Harrasse et al. (2026), cross-layer mechanisms:** *Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders*, EMNLP. Particularly close to cross-layer attribution graphs.
- **Tools:** Neuronpedia (SAE features, steering, attribution graphs), SAELens (SAE training/analysis), TransformerLens (activation cache/hooks), NNsight (model tracing/intervention), Gemma Scope 2 (Gemma SAEs/transcoders).

These precedents mean the broad research territory is established. They do not validate WeightLiftAI's specific dialect or affective findings. Its potential differentiation is a reproducible, local-first workflow joining layer-wise multilingual geometry, culturally reviewed dialect contrasts, affective concepts, lexical counterfactuals, SAE feature grounding, and causal validation in one experiment registry.

## References
- Tak et al. 2025: https://aclanthology.org/2025.findings-acl.679/
- Chang et al. 2022: https://aclanthology.org/2022.emnlp-main.9/
- Havaldar et al. 2023: https://aclanthology.org/2023.wassa-1.19/
- Zhang et al. 2025: https://proceedings.iclr.cc/paper_files/paper/2025/hash/edd00cead3425393baf13004de993017-Abstract-Conference.html
- Ben Tamo et al. 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/00295cede6e1600d344b5cd6d9fd4640-Abstract-Conference.html
- Harrasse et al. 2026: https://aclanthology.org/2026.emnlp-main.2222/
- Neuronpedia: https://docs.neuronpedia.org/
- SAELens: https://github.com/decoderesearch/SAELens
- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- NNsight: https://nnsight.net/
- Gemma Scope 2: https://deepmind.google/models/gemma/gemma-scope/

## Open design questions
- Which capture sites and pooling conventions should be defaults for sentence-level comparisons?
- Should the first interface be notebook-first, static HTML, or a local web app?
- Which model families can be supported reliably on 8 GB consumer GPUs?
- How should dialect review and reviewer disagreement be represented?
- Which causal tests are practical and valid for each architecture?
