# Agent Handoff — WeightLiftAI Research & Product Direction

**Last updated:** 2026-11-04  
**Purpose:** Bring a new coding/research agent up to speed on the project's current state, scientific direction, prior work, and next implementation priorities.

> Start here. Read this document, then the linked research documents before changing code. This is a handoff and plan, not a claim that every proposed interface or feature has been implemented.

## 1. One-sentence mission

WeightLiftAI is a local-first, reproducible laboratory for inspecting, mapping, evaluating, and carefully intervening on open-weight language models — with a research focus on multilingual representations, culturally situated affective language, and eventually medically grounded knowledge.

**Core principle:** Map representations. Trace transformations. Test hypotheses. Preserve evidence.

## 2. Current repository state

The README describes an active local research platform with:
- Model inspection and hardware diagnostics.
- Activation capture through temporary hooks with cleanup.
- Multi-layer SAE encoding/tracing and interactive feature visualizations.
- Reversible SAE feature interventions.
- LoRA/PEFT adaptation comparisons.
- Experiment manifests, an audit ledger, and a rolling research paper.
- Experiments 001 and 002 executed; Experiments 003–005 planned.

Important: verify actual code, tests, artifacts, and claims-register entries before treating README checkboxes or reported metrics as independently validated. Do not silently upgrade a planned capability to implemented status.

### Read these first
1. [README](../README.md) — current project overview and experiment status.
2. [Representation Cartography plan](REPRESENTATION_CARTOGRAPHY.md) — product surfaces, schemas, safeguards, and phases.
3. [Research Roadmap](RESEARCH_ROADMAP.md) — current workstreams.
4. [Rolling Paper](ROLLING_PAPER.md) — methods, claims, confidence intervals, and limitations.
5. [Audit Ledger](AUDIT_LEDGER.jsonl) — machine-readable run/claim history.

## 3. Research focus and why it matters

The distinctive research intersection is:
- Mechanistic interpretability of affective/emotion-related representations.
- Multilingual and dialect-sensitive representation geometry.
- Culturally situated idioms of distress, including Dominican Spanish.
- Matched semantic contrasts and lexical counterfactuals to expose shortcut learning.
- SAE feature grounding and cross-layer causal tests.
- A local-first experiment workflow that preserves provenance and reproducibility.

The key question is not simply whether two sentences have similar embeddings. It is:

**When people express similar distress in different languages, dialects, and cultural contexts, what information is represented, where is it transformed or lost, and which internal pathways causally affect model behavior?**

Illustrative probes (not presumed equivalents; require context and human review):
- “I feel depressed.”
- “Ando bajito.”
- “Tengo los nervios de punta.”
- “Se me aprieta el pecho.”

Do not flatten these into direct translations. Build matched sets with context, pragmatic intent, register, region, and reviewer notes. Treat dialect labels as provenance-rich metadata, not a simplistic categorical truth.

## 4. Prior work to learn from

These papers establish that important parts of the research direction already exist. The opportunity is to build a careful, integrated workflow and test a specific intersection — not to claim the field is untouched.

1. **Tak et al. (2025), “Mechanistic Interpretability of Emotion Inference in Large Language Models,” Findings of ACL.** Emotion representations, appraisal concepts, and causal interventions.  
   https://aclanthology.org/2025.findings-acl.679/
2. **Chang, Tu & Bergen (2022), “The Geometry of Multilingual Language Model Representations,” EMNLP.** Shared and language-sensitive representation subspaces across many languages.  
   https://aclanthology.org/2022.emnlp-main.9/
3. **Havaldar et al. (2023), “Multilingual Language Models are not Multicultural: A Case Study in Emotion,” WASSA/ACL.** Multilingual ability does not guarantee culturally appropriate emotional understanding.  
   https://aclanthology.org/2023.wassa-1.19/
4. **Zhang et al. (2025), “The Same but Different: Structural Similarities and Differences in Multilingual Language Modeling,” ICLR.** Shared and language-specific internal circuits.  
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/edd00cead3425393baf13004de993017-Abstract-Conference.html
5. **Ben Tamo et al. (2026), “LinguaMap: Which Layers of LLMs Speak Your Language and How to Tune Them?”, ICLR.** Layer-wise language representations and selective tuning.  
   https://proceedings.iclr.cc/paper_files/paper/2026/hash/00295cede6e1600d344b5cd6d9fd4640-Abstract-Conference.html
6. **Harrasse et al. (2026), “Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders,” EMNLP.** Cross-layer transcoders, attribution graphs, and multilingual circuit tracing.  
   https://aclanthology.org/2026.emnlp-main.2222/

### Tools and ecosystems to study
- Neuronpedia — SAE feature search, visualization, steering, attribution graphs: https://docs.neuronpedia.org/
- TransformerLens — activation caching and hooks: https://github.com/TransformerLensOrg/TransformerLens
- SAELens — SAE training and analysis: https://github.com/decoderesearch/SAELens
- NNsight — model tracing and interventions: https://nnsight.net/
- Gemma Scope 2 — Gemma SAE/transcoder resources: https://deepmind.google/models/gemma/gemma-scope/

Use existing tested project abstractions where possible. Avoid duplicating activation hooks or building a second experiment registry.

## 5. Medical knowledge: proposed research extension

Medical knowledge belongs in this project, but it should be a **separate, explicit knowledge-grounding track** that connects to the representation work without conflating clinical facts, language, and internal model features.

There are at least three distinct questions:

1. **Knowledge representation:** What medical concepts and relationships does a model encode or retrieve?
2. **Language and access:** Does it understand the same medical concept when expressed in plain language, multilingual language, dialect, abbreviation, or an idiom?
3. **Clinical reasoning and safety:** Does the model use relevant evidence correctly, distinguish uncertainty, avoid unsupported inference, and communicate limitations?

A model can perform well on medical question answering without robust internal medical representations; it can also encode medical terminology without reasoning safely. Evaluate these separately.

### 5.1 Medical Knowledge Atlas

Proposed product surface:
- Browse a versioned medical concept graph.
- Link terms, synonyms, abbreviations, lay expressions, multilingual variants, and coded concepts.
- Represent relations such as symptom-of, associated-with, contraindicated-with, treats, risk-factor-for, and differential-diagnosis-of only when supported by an authoritative source.
- Compare model activations for matched concept expressions across layers and languages.
- Inspect whether retrieved/decoded knowledge is stable across paraphrases, contexts, and model revisions.
- Show source, release/version, license, evidence level, and provenance for every concept/edge.
- Allow expert review and disagreement annotations; do not turn model-generated associations into facts.

### 5.2 Candidate knowledge sources

Investigate licensing, access terms, versioning, and permitted use before integration. Do not scrape or redistribute restricted content.

- **UMLS Metathesaurus:** broad biomedical vocabularies and mappings; access is governed by UMLS licensing and a UMLS account/agreement. Useful as a crosswalk layer, not a single gold-standard ontology. https://www.nlm.nih.gov/research/umls/
- **SNOMED CT:** clinical terminology and relationships; use is subject to territory-specific licensing and release terms. https://www.snomed.org/
- **MeSH:** NLM controlled vocabulary for biomedical literature indexing; useful for topic/concept organization. https://www.nlm.nih.gov/mesh/
- **RxNorm:** normalized clinical drug names and relationships; useful for medication terminology. https://www.nlm.nih.gov/research/umls/rxnorm/
- **ICD-10-CM:** diagnosis classification for coding/reporting; it is not a complete clinical ontology or a substitute for clinical reasoning. https://www.cdc.gov/nchs/icd/icd-10-cm/
- **MedlinePlus:** patient-facing health information and consumer-friendly terminology; useful for lay-language alignment, not a comprehensive structured knowledge graph. https://medlineplus.gov/
- **PubMed / NCBI resources:** literature discovery and metadata; abstracts and articles have separate copyright and reuse terms. https://pubmed.ncbi.nlm.nih.gov/
- **Open access clinical guidelines:** only ingest when the specific publisher/organization license permits the intended use; store citation and version metadata.

Potential future additions (evaluate carefully): Human Phenotype Ontology, Orphanet, Disease Ontology, and open clinical guideline datasets. Verify each source's license and coverage before committing.

### 5.3 Proposed medical data model

Start with a small, source-agnostic schema. Keep identifiers namespaced by source and release.

- `MedicalConcept`: source, source_id, release, preferred_label, definition, semantic_type, language, license, provenance.
- `ConceptSynonym`: concept_ref, term, language, dialect/region, term_type (synonym, abbreviation, lay term, historical term), provenance, reviewer_status.
- `ConceptRelation`: subject_ref, predicate, object_ref, source, evidence/provenance, release, confidence/evidence_grade where source-defined.
- `ClinicalExpression`: concept_ref, expression, language, locale, context, register, intended meaning, ambiguity notes, human reviewer.
- `KnowledgeProbe`: probe_id, target concepts/relations, expected evidence, distractors, ambiguity, split, author/reviewer.
- `KnowledgeEvaluationRun`: model/revision, prompt/template, retrieval configuration, data release/hash, metrics, uncertainty, errors, safety checks, artifact references.

Keep medical facts, linguistic expressions, model representations, and model outputs as separate entities linked by stable IDs.

### 5.4 Evaluation suite

Build a benchmark with separate slices:
- Terminology recognition: canonical term, synonym, abbreviation, lay expression.
- Cross-lingual equivalence: meaning-preserving expressions across languages, with expert/native-speaker review.
- Relation understanding: identify supported relationships and reject unsupported ones.
- Context sensitivity: distinguish same term across clinical contexts and senses.
- Evidence-grounded QA: answer only from provided/versioned sources and cite the evidence.
- Calibration: abstention and uncertainty when evidence is missing or ambiguous.
- Contradiction handling: detect conflicting sources and report dates/versions.
- Safety: distinguish educational information from diagnosis or treatment; avoid inventing certainty.
- Robustness: typos, misspellings, speech-like text, code-switching, dialect, and paraphrases.

Metrics should include concept/relation precision and recall, exact/semantic match where appropriate, calibration, abstention quality, citation support, subgroup performance, and paired confidence intervals. Avoid one aggregate score hiding clinically important failures.

### 5.5 Mechanistic questions

Once the knowledge benchmark is stable:
- At which layers are medical concepts and relations decodable?
- Are canonical terms and lay expressions represented similarly or only connected by context?
- Do multilingual expressions converge on shared representations, and where do language-specific signals remain?
- Do medical terms trigger spurious associations from lexical overlap?
- Can SAE features be grounded against curated concept sets without assuming one feature equals one concept?
- Do retrieval-augmented answers improve factual grounding while internal representations remain unchanged?
- Do fine-tuning or steering interventions improve held-out performance without harming uncertainty calibration or unrelated capabilities?

Compare base model, retrieval-augmented model, and adapted model separately. RAG can improve answer access without changing the model's internal knowledge; do not describe RAG gains as representation changes.

### 5.6 Medical safeguards

- No patient records, protected health information, or identifiable clinical conversations in exploratory experiments.
- Prefer public, synthetic, appropriately de-identified, or explicitly authorized data with documented governance.
- Do not use generated labels as medical ground truth.
- Clinical expert review is required for medical claims and benchmark adjudication.
- Track source release, date, jurisdiction, and evidence level; medical knowledge changes.
- No diagnosis, triage, prescribing, treatment recommendation, or clinical decision support deployment from this research platform.
- Do not infer a person's condition from dialect, emotion wording, or latent features.
- Keep the medical knowledge track research-only unless a separate regulatory, privacy, security, and clinical governance program is established.

## 6. Suggested implementation sequence

Do not start by building a giant medical knowledge graph or a complex UI. Start with data contracts and a tiny, auditable vertical slice.

### Immediate priorities
1. **Audit the current claims and experiment artifacts.** Reconcile the noted Experiment 002b feature #2323 discrepancy; replace zero-denominator ratios (for example #602 ratio 999.0) with an explicit undefined/denominator-zero representation. Report empirical p-values at finite resolution (e.g. if no random control is more extreme among 20, report the corresponding bound, not p=0).
2. **Finish Phase 0 schemas for Representation Cartography.** Matched-item dataset manifest, capture spec, artifact versioning, masking/pooling contract.
3. **Build Layer Atlas first.** Reuse existing capture code; one command should capture selected layers and produce original-space metrics plus provenance.
4. **Add deterministic PCA map after artifact schema stabilizes.** Never use projection coordinates as scientific metrics.
5. **Build a medical terminology pilot as a separate dataset adapter.** Start with a small permitted source (for example MeSH or RxNorm subset, after confirming terms) and 50–100 reviewed concepts, not an unbounded ingest.
6. **Create paired probes:** canonical medical term vs. plain-language expression vs. reviewed multilingual expression. Keep concept ID and context fixed.
7. **Evaluate before visualizing.** Establish baselines, splits, uncertainty, source citations, and error taxonomy.
8. **Only then connect medical probes to Layer Atlas / Feature Observatory.**

### Definition of done for first medical pilot
- Source license and version recorded.
- 50–100 concepts have stable source IDs and provenance.
- Synonyms/lay expressions have language and reviewer metadata.
- Every probe has a target concept, context, expected evidence, and split assignment.
- No PHI or patient data is present.
- Baseline evaluation is reproducible and reports uncertainty and unsupported-answer behavior.
- A model answer cannot silently create or alter a knowledge-graph fact.

## 7. Scientific rules that apply across all tracks

- Projection is not geometry; use original-space metrics for quantitative claims.
- Decodability is not use.
- Correlation is not causation.
- SAE feature IDs are not neuron indices.
- A multilingual model is not automatically multicultural.
- Translation equivalence must be validated in context.
- A clinical code is not a full clinical concept.
- Retrieval success is not evidence that knowledge is stored internally.
- Correlation between an activation and a medical concept is not clinical validity.
- Record negative results, uncertainty, failed replications, and withdrawn claims.
- Pin model revisions, tokenizer, data release/hash, code commit, software, hardware, seeds, and capture settings.

## 8. Agent operating instructions

Before coding:
1. Inspect the repository tree, active branch, recent commits, tests, and current docs.
2. Read the five handoff documents listed above.
3. Identify which roadmap item the change advances.
4. Write or update a short issue/plan with acceptance criteria.
5. Reuse existing modules and schemas unless there is a clear reason not to.

During coding:
- Keep changes small and testable.
- Do not silently rewrite scientific claims or historical results.
- Do not store raw text/activations by default.
- Never overwrite base model weights.
- Add tests for masking, deterministic behavior, variable lengths, hook cleanup, artifact integrity, and privacy defaults as relevant.
- Make source licenses and provenance explicit for medical data.

Before finishing:
- Run relevant tests and report exact commands/results.
- Update README/roadmap only for actually completed work.
- Record commit, model/data versions, metrics, limitations, and artifacts.
- Clearly separate implemented, experimental, proposed, and blocked work.

## 9. The next concrete task

**Implement the Phase 0 Representation Cartography schemas and a small medical terminology pilot contract — not the frontend.**

Deliver:
- Versioned JSON/Pydantic schemas for the data objects above.
- A validation command for matched multilingual/medical probe datasets.
- A tiny fixture dataset with synthetic or public non-sensitive examples and source provenance.
- Tests for stable IDs, duplicate concepts, missing provenance, language metadata, leakage between train/validation/test, and invalid relations.
- A short design decision record explaining why medical knowledge is a separate track connected through shared probe/representation IDs.

Do not ingest large medical corpora or expose a clinical-facing feature in this task.
