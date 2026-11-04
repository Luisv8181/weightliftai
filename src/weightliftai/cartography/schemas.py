"""Representation Cartography & Knowledge Grounding Schemas.

Versioned data models defining:
1. ModelRef, DatasetRef, CaptureSpec, RepresentationArtifact, ProjectionArtifact, ComparisonRun
2. MedicalConcept, ConceptSynonym, ConceptRelation, ClinicalExpression, KnowledgeProbe

Follows the design specifications in docs/REPRESENTATION_CARTOGRAPHY.md and docs/AGENT_HANDOFF.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ===========================================================================
# 1. Representation Cartography Core Schemas
# ===========================================================================

class ModelRef(BaseModel):
    """Immutable reference to an evaluated model checkpoint and runtime environment."""
    model_config = ConfigDict(frozen=True)

    model_id: str = Field(..., description="Hugging Face model ID or local directory path.")
    revision: str = Field(..., description="Exact git commit hash or immutable version tag.")
    tokenizer_revision: str | None = Field(default=None, description="Tokenizer commit hash if distinct.")
    architecture: str = Field(..., description="PyTorch model class name (e.g., Gemma3ForCausalLM).")
    dtype: str = Field(..., description="Torch dtype used during capture (e.g., float32, bfloat16).")
    device: str | None = Field(default=None, description="Execution device (e.g., cuda:0, cpu).")


class DatasetRef(BaseModel):
    """Metadata and integrity hash for a versioned research probe dataset."""
    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(..., description="Short identifier for the probe dataset.")
    version: str = Field(..., description="Dataset release version (e.g., v0.1, v1.0).")
    sha256: str = Field(..., description="Hex SHA-256 digest of the canonical dataset file.")
    total_items: int = Field(..., ge=1, description="Total number of probe rows.")
    languages: list[str] = Field(..., min_length=1, description="List of ISO language codes (e.g., ['en', 'es']).")
    dialects: list[str] = Field(default_factory=list, description="Regional or dialect variants (e.g., ['es_do']).")
    provenance_notes: str = Field(..., description="Dataset construction methodology and origin.")
    is_synthetic: bool = Field(default=False, description="Whether items were model-generated.")
    has_human_review: bool = Field(default=False, description="Whether items received expert native-speaker review.")


class CaptureSpec(BaseModel):
    """Detailed specification of representation capture site, masking, and pooling."""
    model_config = ConfigDict(frozen=True)

    module_path: str = Field(..., description="Exact submodule path (e.g., model.layers.13).")
    layer_index: int = Field(..., ge=0, description="0-indexed depth in the transformer stack.")
    site: Literal["residual", "attention", "mlp", "sae"] = Field(
        ..., description="Representation extraction site."
    )
    token_mask: Literal["content", "all", "last"] = Field(
        default="content", description="Token inclusion policy (excluding BOS/padding when 'content')."
    )
    pooling: Literal["mean", "last", "max"] = Field(
        default="mean", description="Pooling method over the token sequence dimension."
    )
    normalization: Literal["none", "unit_l2"] = Field(
        default="none", description="Post-pooling vector normalization."
    )
    sae_release: str | None = Field(default=None, description="SAE release identifier if site is 'sae'.")
    sae_id: str | None = Field(default=None, description="SAE dictionary ID within release.")

    @model_validator(mode="after")
    def validate_sae_requirement(self) -> CaptureSpec:
        if self.site == "sae" and not self.sae_id:
            raise ValueError("sae_id must be provided when site is 'sae'")
        return self


class RepresentationArtifact(BaseModel):
    """Container for extracted high-dimensional representations with immutable provenance."""
    model_config = ConfigDict(frozen=True)

    artifact_version: str = Field(default="v1", description="Schema version.")
    model_ref: ModelRef
    dataset_ref: DatasetRef
    capture_spec: CaptureSpec
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    item_ids: list[int | str] = Field(..., min_length=1, description="Item IDs corresponding to rows.")
    feature_dim: int = Field(..., ge=1, description="Dimension of each extracted representation vector.")
    storage_uri: str | None = Field(default=None, description="Path to on-disk array (e.g., .npy, .safetensors).")


class ProjectionArtifact(BaseModel):
    """2D or 3D coordinate projection of representation artifacts for visual exploration."""
    model_config = ConfigDict(frozen=True)

    artifact_version: str = Field(default="v1", description="Schema version.")
    source_artifact_sha256: str = Field(..., description="SHA-256 of the source representation artifact.")
    algorithm: Literal["pca", "umap", "tsne"] = Field(..., description="Dimensionality reduction method.")
    n_components: int = Field(default=2, ge=2, le=3, description="2D or 3D output coordinates.")
    random_seed: int = Field(default=42, description="Random seed used for deterministic projection.")
    explained_variance_ratio: list[float] | None = Field(
        default=None, description="Fraction of variance explained per component (for PCA)."
    )
    coordinates: list[list[float]] = Field(..., description="List of [x, y] or [x, y, z] projected points.")
    item_ids: list[int | str] = Field(..., description="Matching item IDs for each coordinate row.")
    distortion_metrics: dict[str, Any] | None = Field(
        default=None, description="Distortion metrics (trustworthiness, continuity, k-NN overlap)."
    )


class ComparisonRun(BaseModel):
    """Quantitative evaluation and statistical comparison between two representation subsets."""
    model_config = ConfigDict(frozen=True)

    run_id: str
    metric: str = Field(..., description="Name of the computed metric (e.g., cosine_similarity, l2_distance).")
    group_a: str = Field(..., description="Label of primary comparison condition.")
    group_b: str = Field(..., description="Label of contrast comparison condition.")
    estimate: float = Field(..., description="Central point estimate.")
    ci_95: tuple[float, float] | None = Field(default=None, description="95% confidence interval.")
    bootstrap_iterations: int = Field(default=0, ge=0, description="Number of bootstrap resamples if applicable.")
    p_value_bound: str | None = Field(default=None, description="Reported p-value with finite resolution bound.")


# ===========================================================================
# 2. Medical Knowledge Grounding Schemas
# ===========================================================================

class MedicalConcept(BaseModel):
    """Authoritative medical or clinical concept definition from a versioned ontology."""
    model_config = ConfigDict(frozen=True)

    concept_id: str = Field(..., description="Namespaced concept identifier (e.g., MESH:D003866).")
    source: str = Field(..., description="Authoritative vocabulary (e.g., MeSH, RxNorm, SNOMED_CT, ICD10_CM).")
    source_id: str = Field(..., description="Native ID within the source system.")
    release: str = Field(..., description="Source release or edition (e.g., 2026_01, v2026AA).")
    preferred_label: str = Field(..., description="Canonical clinical preferred term.")
    definition: str | None = Field(default=None, description="Authoritative definition from source.")
    semantic_type: str = Field(..., description="High-level category (e.g., Disease or Syndrome, Finding).")
    language: str = Field(default="en", description="ISO language of the preferred label.")
    license: str = Field(..., description="Data reuse terms (e.g., Public Domain, UMLS License).")
    provenance: str = Field(..., description="URI or exact document reference for traceability.")


class ConceptSynonym(BaseModel):
    """Synonym, lay phrase, or cultural expression linked to an authoritative concept."""
    model_config = ConfigDict(frozen=True)

    concept_ref: str = Field(..., description="Foreign key to MedicalConcept.concept_id.")
    term: str = Field(..., description="Alternative surface form or expression.")
    language: str = Field(..., description="ISO 639-1 language code.")
    dialect: str | None = Field(default=None, description="Specific regional dialect code (e.g., es_DO, en_US).")
    term_type: Literal[
        "canonical", "synonym", "abbreviation", "lay_expression", "cultural_idiom"
    ] = Field(..., description="Linguistic relationship to the concept.")
    provenance: str = Field(..., description="Source citation or documentation of this expression.")
    reviewer_status: Literal["unreviewed", "reviewed", "contested"] = Field(
        default="unreviewed", description="Status of expert native-speaker/clinical review."
    )
    reviewer_notes: str | None = Field(default=None, description="Expert clinical/linguistic review comments.")


class ConceptRelation(BaseModel):
    """Supported semantic relationship between two authoritative medical concepts."""
    model_config = ConfigDict(frozen=True)

    subject_ref: str = Field(..., description="Foreign key to subject MedicalConcept.concept_id.")
    predicate: Literal[
        "symptom_of",
        "associated_with",
        "contraindicated_with",
        "treats",
        "risk_factor_for",
        "differential_diagnosis_of",
    ] = Field(..., description="Semantic relation.")
    object_ref: str = Field(..., description="Foreign key to object MedicalConcept.concept_id.")
    source: str = Field(..., description="Authoritative medical guideline or database establishing the link.")
    evidence_grade: str | None = Field(default=None, description="Clinical evidence grade (e.g., Grade A, Expert Consensus).")
    release: str = Field(..., description="Version of the source relationship.")


class ClinicalExpression(BaseModel):
    """Naturalistic utterance or clinical sentence expressing a target concept."""
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    concept_ref: str = Field(..., description="Target MedicalConcept.concept_id.")
    expression: str = Field(..., description="Sentence or conversational phrase.")
    language: str = Field(..., description="ISO 639-1 language code.")
    locale: str | None = Field(default=None, description="Locale or dialect context (e.g., es-DO).")
    speech_register: Literal["clinical", "formal", "colloquial", "idiomatic"] = Field(
        ..., alias="register", description="Sociolinguistic register."
    )
    intended_meaning: str = Field(..., description="Pragmatic intended clinical meaning.")
    ambiguity_notes: str | None = Field(default=None, description="Clinical ambiguity or distractor risks.")
    reviewer: str | None = Field(default=None, description="Name or role of reviewing clinician/linguist.")


class KnowledgeProbe(BaseModel):
    """Unit of evaluation probing model internal representation or retrieval of a concept."""
    model_config = ConfigDict(frozen=True)

    probe_id: str = Field(..., description="Unique immutable probe ID.")
    concept_ref: str = Field(..., description="Target MedicalConcept.concept_id.")
    text: str = Field(..., description="Sentence or prompt text provided to the model.")
    language: str = Field(..., description="ISO 639-1 language code.")
    variant: str = Field(..., description="Condition variant (e.g., canonical, lay_term, cultural_idiom).")
    split: Literal["train", "validation", "test"] = Field(..., description="Partition split assignment.")
    expected_label: str = Field(..., description="Ground truth concept name or answer.")
    distractors: list[str] = Field(default_factory=list, description="Candidate plausible distractors.")
    reviewer_status: Literal["unreviewed", "reviewed", "contested"] = Field(default="unreviewed")
