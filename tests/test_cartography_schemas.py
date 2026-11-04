"""Tests for Representation Cartography and Medical Knowledge schemas & validators."""

import pandas as pd
import pytest

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


def test_model_ref_immutability():
    m = ModelRef(
        model_id="google/gemma-3-1b-it",
        revision="dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        architecture="Gemma3ForCausalLM",
        dtype="float32",
        device="cuda:0",
    )
    assert m.model_id == "google/gemma-3-1b-it"
    with pytest.raises(Exception):
        m.model_id = "other"  # frozen


def test_capture_spec_sae_validation():
    # Residual site does not require sae_id
    spec_res = CaptureSpec(module_path="model.layers.13", layer_index=13, site="residual")
    assert spec_res.site == "residual"

    # SAE site without sae_id must raise ValueError
    with pytest.raises(ValueError, match="sae_id must be provided"):
        CaptureSpec(module_path="model.layers.13", layer_index=13, site="sae")

    # SAE site with sae_id succeeds
    spec_sae = CaptureSpec(
        module_path="model.layers.13",
        layer_index=13,
        site="sae",
        sae_release="gemma-scope-2-1b-it-res",
        sae_id="layer_13_width_16k_l0_medium",
    )
    assert spec_sae.sae_id == "layer_13_width_16k_l0_medium"


def test_validate_matched_dataset_success():
    data = [
        {"item_id": 1, "variant": "en", "lang": "en", "text": "I feel very down."},
        {"item_id": 1, "variant": "es_neutral", "lang": "es", "text": "Me siento muy desanimado."},
        {"item_id": 1, "variant": "es_do", "lang": "es", "text": "Ando con el ánimo en el piso."},
        {"item_id": 2, "variant": "en", "lang": "en", "text": "A heavy sadness."},
        {"item_id": 2, "variant": "es_neutral", "lang": "es", "text": "Una tristeza pesada."},
        {"item_id": 2, "variant": "es_do", "lang": "es", "text": "Una pena honda que no se me quita."},
    ]
    df = pd.DataFrame(data)
    res = validate_matched_dataset(df, expected_variants=["en", "es_neutral", "es_do"])
    assert res["status"] == "valid"
    assert res["unique_items"] == 2
    assert res["total_rows"] == 6


def test_validate_matched_dataset_missing_column():
    df = pd.DataFrame([{"item_id": 1, "text": "hello"}])
    with pytest.raises(ValidationError, match="missing required column"):
        validate_matched_dataset(df)


def test_validate_matched_dataset_empty_text():
    df = pd.DataFrame([
        {"item_id": 1, "variant": "en", "lang": "en", "text": "Valid"},
        {"item_id": 1, "variant": "es", "lang": "es", "text": "   "},
    ])
    with pytest.raises(ValidationError, match="Found empty text"):
        validate_matched_dataset(df)


def test_validate_matched_dataset_unmatched_variant():
    df = pd.DataFrame([
        {"item_id": 1, "variant": "en", "lang": "en", "text": "Sentence 1"},
        {"item_id": 1, "variant": "es", "lang": "es", "text": "Oración 1"},
        {"item_id": 2, "variant": "en", "lang": "en", "text": "Sentence 2"},
        # item 2 is missing "es"
    ])
    with pytest.raises(ValidationError, match="Unmatched items detected"):
        validate_matched_dataset(df, expected_variants=["en", "es"])


def test_validate_matched_dataset_duplicate_pairs():
    df = pd.DataFrame([
        {"item_id": 1, "variant": "en", "lang": "en", "text": "First"},
        {"item_id": 1, "variant": "en", "lang": "en", "text": "Duplicate"},
    ])
    with pytest.raises(ValidationError, match="Duplicate"):
        validate_matched_dataset(df)


def test_medical_concept_and_probe_validation():
    concept = MedicalConcept(
        concept_id="MESH:D003866",
        source="MeSH",
        source_id="D003866",
        release="2026_01",
        preferred_label="Depressive Disorder",
        semantic_type="Mental or Behavioral Dysfunction",
        language="en",
        license="Public Domain (NLM)",
        provenance="https://id.nlm.nih.gov/mesh/D003866",
    )

    probes = [
        KnowledgeProbe(
            probe_id="probe_001",
            concept_ref="MESH:D003866",
            text="The patient exhibits persistent feelings of worthlessness and depressive disorder.",
            language="en",
            variant="canonical_clinical",
            split="train",
            expected_label="Depressive Disorder",
        ),
        KnowledgeProbe(
            probe_id="probe_002",
            concept_ref="MESH:D003866",
            text="Lleva meses sintiendo una pena honda y un desánimo que no se le quita.",
            language="es",
            variant="cultural_idiom",
            split="test",
            expected_label="Depressive Disorder",
        ),
    ]

    res = validate_knowledge_probes(probes, [concept])
    assert res["status"] == "valid"
    assert res["total_probes"] == 2


def test_validate_knowledge_probes_data_leakage():
    concept = MedicalConcept(
        concept_id="MESH:D003866",
        source="MeSH",
        source_id="D003866",
        release="2026_01",
        preferred_label="Depressive Disorder",
        semantic_type="Mental or Behavioral Dysfunction",
        language="en",
        license="Public Domain (NLM)",
        provenance="https://id.nlm.nih.gov/mesh/D003866",
    )

    leak_text = "The patient describes acute chest pain and shortness of breath."
    probes = [
        KnowledgeProbe(
            probe_id="probe_001",
            concept_ref="MESH:D003866",
            text=leak_text,
            language="en",
            variant="canonical",
            split="train",
            expected_label="Depressive Disorder",
        ),
        KnowledgeProbe(
            probe_id="probe_002",
            concept_ref="MESH:D003866",
            text=leak_text,  # Identical text across train and test!
            language="en",
            variant="canonical",
            split="test",
            expected_label="Depressive Disorder",
        ),
    ]

    with pytest.raises(ValidationError, match="Data leakage detected"):
        validate_knowledge_probes(probes, [concept])


def test_validate_concept_relations_self_referential():
    c = MedicalConcept(
        concept_id="MESH:D003866",
        source="MeSH",
        source_id="D003866",
        release="2026_01",
        preferred_label="Depressive Disorder",
        semantic_type="Mental or Behavioral Dysfunction",
        language="en",
        license="Public Domain (NLM)",
        provenance="https://id.nlm.nih.gov/mesh/D003866",
    )
    rel = ConceptRelation(
        subject_ref="MESH:D003866",
        predicate="associated_with",
        object_ref="MESH:D003866",  # Self-referential!
        source="Test Guideline",
        release="2026",
    )
    with pytest.raises(ValidationError, match="Self-referential relation detected"):
        validate_concept_relations([rel], [c])
