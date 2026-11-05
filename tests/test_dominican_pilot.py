"""Tests for the Dominican Spanish Pilot dataset."""

import pytest

from weightliftai.cartography import DominicanPilotDataset


def test_dominican_pilot_loading_and_counts():
    ds = DominicanPilotDataset()
    assert len(ds.items) == 50
    assert ds.validation_report["status"] == "valid"
    assert ds.validation_report["total_concepts"] == 50
    assert ds.validation_report["splits"] == {"train": 30, "val": 10, "test": 10}
    assert ds.validation_report["unique_categories"] >= 10
    assert ds.dataset_ref.dataset_id == "dominican_spanish_pilot_50"
    assert len(ds.dataset_ref.sha256) == 64


def test_dominican_pilot_to_matched_dataframe():
    ds = DominicanPilotDataset()
    
    # All splits: 50 concepts * 3 variants = 150 rows
    df_all = ds.to_matched_dataframe()
    assert len(df_all) == 150
    assert set(df_all["variant"].unique()) == {"dominican_spanish", "neutral_spanish", "english_pragmatic"}

    # Test split: 10 concepts * 3 variants = 30 rows
    df_test = ds.to_matched_dataframe(split="test")
    assert len(df_test) == 30
    assert all(df_test["split"] == "test")

    # Check required columns
    required_cols = {"pair_id", "concept_id", "text", "variant", "language", "dialect", "split", "negative_control"}
    assert required_cols.issubset(df_test.columns)
    assert not df_test["text"].isna().any()
    assert not (df_test["text"] == "").any()
