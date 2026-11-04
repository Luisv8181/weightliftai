"""Unit tests for Representation Map dimensionality reduction, distortion metrics, and inspection data."""

import numpy as np
import pandas as pd
import pytest

from weightliftai.cartography.representation_map import (
    build_point_inspection_data,
    compute_distortion_metrics,
    generate_map_html,
    project_representations,
)
from weightliftai.cartography.schemas import ProjectionArtifact


def test_compute_distortion_metrics():
    rng = np.random.default_rng(42)
    high_dim = rng.normal(size=(30, 32))
    low_dim = high_dim[:, :2]

    metrics = compute_distortion_metrics(high_dim, low_dim, k_neighbors=5)
    assert "trustworthiness" in metrics
    assert "continuity" in metrics
    assert "knn_overlap" in metrics
    assert "distortion_warning" in metrics
    assert "interpretive_note" in metrics

    assert 0.0 <= metrics["trustworthiness"] <= 1.0
    assert 0.0 <= metrics["continuity"] <= 1.0
    assert 0.0 <= metrics["knn_overlap"] <= 1.0
    assert isinstance(metrics["distortion_warning"], bool)

    # Edge case: very small sample
    small_m = compute_distortion_metrics(high_dim[:2], low_dim[:2])
    assert small_m["trustworthiness"] == 1.0
    assert small_m["distortion_warning"] is False


def test_project_representations_pca():
    rng = np.random.default_rng(123)
    embeddings = rng.normal(size=(25, 64))
    item_ids = [f"item_{i}" for i in range(25)]

    artifact, coords = project_representations(
        embeddings,
        item_ids,
        algorithm="pca",
        n_components=2,
        random_seed=42,
        k_neighbors=4,
    )

    assert isinstance(artifact, ProjectionArtifact)
    assert artifact.algorithm == "pca"
    assert artifact.n_components == 2
    assert len(artifact.coordinates) == 25
    assert len(artifact.coordinates[0]) == 2
    assert coords.shape == (25, 2)
    assert artifact.explained_variance_ratio is not None
    assert len(artifact.explained_variance_ratio) == 2
    assert sum(artifact.explained_variance_ratio) <= 1.0001
    assert artifact.distortion_metrics is not None
    assert "trustworthiness" in artifact.distortion_metrics


def test_project_representations_tsne():
    rng = np.random.default_rng(456)
    embeddings = rng.normal(size=(20, 16))
    item_ids = [f"id_{i}" for i in range(20)]

    artifact, coords = project_representations(
        embeddings,
        item_ids,
        algorithm="tsne",
        n_components=2,
        random_seed=99,
        perplexity=5,
    )

    assert artifact.algorithm == "tsne"
    assert coords.shape == (20, 2)
    assert artifact.explained_variance_ratio is None
    assert artifact.distortion_metrics is not None


def test_project_representations_validation_errors():
    embeddings = np.zeros((10, 8))
    # Mismatch length
    with pytest.raises(ValueError, match="Mismatch"):
        project_representations(embeddings, ["1", "2"])

    # Invalid n_components
    with pytest.raises(ValueError, match="n_components must be 2 or 3"):
        project_representations(embeddings, [str(i) for i in range(10)], n_components=4)

    # Invalid algorithm
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        project_representations(embeddings, [str(i) for i in range(10)], algorithm="invalid")


def test_build_point_inspection_data():
    df = pd.DataFrame([
        {"item_id": "D01", "concept": "Hypertension", "variant": "canonical_clinical", "lang": "en", "split": "test", "text": "Essential hypertension diagnosis"},
        {"item_id": "D01", "concept": "Hypertension", "variant": "plain_lay", "lang": "en", "split": "test", "text": "High blood pressure condition"},
        {"item_id": "D01", "concept": "Hypertension", "variant": "cultural_idiom", "lang": "es", "split": "test", "text": "Presión alta / tensión"},
        {"item_id": "D02", "concept": "Stroke", "variant": "canonical_clinical", "lang": "en", "split": "test", "text": "Cerebrovascular accident"},
        {"item_id": "D02", "concept": "Stroke", "variant": "plain_lay", "lang": "en", "split": "test", "text": "Stroke event"},
        {"item_id": "D02", "concept": "Stroke", "variant": "cultural_idiom", "lang": "es", "split": "test", "text": "Derrame cerebral / trombosis"},
    ])

    rng = np.random.default_rng(789)
    embeddings = rng.normal(size=(6, 16))
    coordinates = rng.normal(size=(6, 2))

    points = build_point_inspection_data(df, embeddings, coordinates, k_neighbors=3)
    assert len(points) == 6

    # Verify first point structure
    p0 = points[0]
    assert p0["item_id"] == "D01"
    assert p0["concept"] == "Hypertension"
    assert p0["variant"] == "canonical_clinical"
    assert len(p0["nearest_neighbors"]) == 3
    assert all("cosine" in nn for nn in p0["nearest_neighbors"])

    # Matched partners: D01 has plain_lay and cultural_idiom
    assert len(p0["matched_partners"]) == 2
    partner_variants = {mp["variant"] for mp in p0["matched_partners"]}
    assert partner_variants == {"plain_lay", "cultural_idiom"}
    assert all("high_dim_cosine" in mp for mp in p0["matched_partners"])
    assert all("low_dim_euclidean" in mp for mp in p0["matched_partners"])


def test_generate_map_html():
    df = pd.DataFrame([
        {"item_id": "D01", "concept": "Hypertension", "variant": "canonical_clinical", "lang": "en", "split": "test", "text": "Hypertension"},
        {"item_id": "D01", "concept": "Hypertension", "variant": "plain_lay", "lang": "en", "split": "test", "text": "High BP"},
    ])
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1]])
    artifact, coords = project_representations(embeddings, ["D01", "D01"], algorithm="pca", random_seed=42)
    points = build_point_inspection_data(df, embeddings, coords, k_neighbors=1)

    html_out = generate_map_html(
        model_name="google/gemma-3-1b-it",
        dataset_name="Medical Pilot",
        layer=13,
        projection_artifact=artifact,
        points_data=points,
        depth_pct=52.0,
        title="Test Map Title",
    )

    assert "<!DOCTYPE html>" in html_out
    assert "Test Map Title" in html_out
    assert "google/gemma-3-1b-it" in html_out
    assert "Layer: <strong>13" in html_out
    assert "Trustworthiness T(k)" in html_out
    assert "Projections are exploratory visuals, not metric geometry" in html_out
    assert "scatterCanvas" in html_out
    assert "inspectorPanel" in html_out
