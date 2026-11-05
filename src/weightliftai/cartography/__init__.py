"""Representation Cartography & Knowledge Grounding module."""

from weightliftai.cartography.dominican_pilot import DominicanPilotDataset
from weightliftai.cartography.jacobian_lens import (
    aggregate_jacobian_corpus,
    check_finite_differences,
    compare_logit_lens_and_jacobian_lens,
    compute_vjp,
    get_unembedding_weight,
)
from weightliftai.cartography.layer_atlas import (
    compute_layer_metrics,
    content_token_mask,
    extract_layer_representations,
    find_decoder_layers,
    generate_atlas_html,
)
from weightliftai.cartography.medical_pilot import MedicalPilotDataset
from weightliftai.cartography.representation_map import (
    build_point_inspection_data,
    compute_distortion_metrics,
    generate_map_html,
    project_representations,
)
from weightliftai.cartography.schemas import (
    CaptureSpec,
    ClinicalExpression,
    ComparisonRun,
    ConceptRelation,
    ConceptSynonym,
    DatasetRef,
    KnowledgeProbe,
    MedicalConcept,
    ModelRef,
    ProjectionArtifact,
    RepresentationArtifact,
)
from weightliftai.cartography.validators import (
    ValidationError,
    validate_concept_relations,
    validate_knowledge_probes,
    validate_matched_dataset,
)

__all__ = [
    "CaptureSpec",
    "ClinicalExpression",
    "ComparisonRun",
    "ConceptRelation",
    "ConceptSynonym",
    "DatasetRef",
    "DominicanPilotDataset",
    "KnowledgeProbe",
    "MedicalConcept",
    "MedicalPilotDataset",
    "ModelRef",
    "ProjectionArtifact",
    "RepresentationArtifact",
    "ValidationError",
    "aggregate_jacobian_corpus",
    "build_point_inspection_data",
    "check_finite_differences",
    "compare_logit_lens_and_jacobian_lens",
    "compute_distortion_metrics",
    "compute_layer_metrics",
    "compute_vjp",
    "content_token_mask",
    "extract_layer_representations",
    "find_decoder_layers",
    "generate_atlas_html",
    "generate_map_html",
    "get_unembedding_weight",
    "project_representations",
    "validate_concept_relations",
    "validate_knowledge_probes",
    "validate_matched_dataset",
]

