"""Tests for the Medical Terminology Pilot dataset adapter."""

import pytest

from weightliftai.cartography import MedicalPilotDataset


def test_medical_pilot_loading_and_counts():
    ds = MedicalPilotDataset()
    assert len(ds.concepts) == 50
    assert len(ds.synonyms) == 150
    assert len(ds.probes) == 150
    assert ds.validation_report["status"] == "valid"
    assert ds.validation_report["unique_concepts_probed"] == 50
    assert ds.validation_report["leakage_detected"] is False


def test_medical_pilot_probes_filtering():
    ds = MedicalPilotDataset()
    train_probes = ds.get_probes(split="train")
    val_probes = ds.get_probes(split="validation")
    test_probes = ds.get_probes(split="test")

    assert len(train_probes) == 90  # 30 concepts * 3 variants
    assert len(val_probes) == 30    # 10 concepts * 3 variants
    assert len(test_probes) == 30   # 10 concepts * 3 variants

    # Language filters
    es_probes = ds.get_probes(language="es")
    assert len(es_probes) == 50  # 1 cultural idiom per concept
    assert all(p.variant == "cultural_idiom" for p in es_probes)

    en_canonical = ds.get_probes(language="en", variant="canonical_clinical")
    assert len(en_canonical) == 50


def test_medical_pilot_to_dataframe_matched_validation():
    ds = MedicalPilotDataset()
    df = ds.to_dataframe()
    assert len(df) == 150
    assert set(df["variant"].unique()) == {"canonical_clinical", "plain_lay", "cultural_idiom"}
    assert set(df["lang"].unique()) == {"en", "es"}
    assert df["item_id"].nunique() == 50


def test_medical_pilot_provenance_and_licensing():
    ds = MedicalPilotDataset()
    for c in ds.concepts:
        assert c.source == "MeSH"
        assert c.release == "2026_01"
        assert "Public Domain" in c.license
        assert c.concept_id.startswith("MESH:D")

    for s in ds.synonyms:
        assert s.reviewer_status == "reviewed"
        assert s.language in ["en", "es"]
