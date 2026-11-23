"""Tests for Affective Cartography Diagnostic Pipeline and THX-02 Two-Stage Repair Framework."""

import pytest
import torch
import torch.nn as nn

from weightliftai.cartography import (
    AffectiveCartographyCorpus,
    AffectiveCartographyPipeline,
    AffectiveCorpusItem,
    AffectiveJacobianTracer,
    SubspaceProjectionSolver,
    THX02CircuitDiagnoser,
)


class DummyDecoderBlock(nn.Module):
    """Simple linear-layer decoder block mimicking transformer layer."""

    def __init__(self, d_model: int = 32):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DummyDecoderModel(nn.Module):
    """Multi-layer dummy decoder model."""

    def __init__(self, num_layers: int = 6, d_model: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderBlock(d_model) for _ in range(num_layers)])
        self.lm_head = nn.Linear(d_model, 100, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(h)


def test_affective_corpus_initialization_and_filtering():
    corpus = AffectiveCartographyCorpus()
    assert len(corpus.items) >= 8

    idioms = corpus.get_by_category("dominican_idiom")
    assert len(idioms) >= 4
    assert any("qué lo qué" in item.text for item in idioms)
    assert any("Tranquilo como una foto" in item.text for item in idioms)

    crisis = corpus.get_by_category("crisis_signal")
    assert len(crisis) >= 3
    assert any("explotar" in item.text for item in crisis)
    assert any("desaparecer" in item.text for item in crisis)

    controls = corpus.get_by_category("neutral_control")
    assert len(controls) >= 1


def test_affective_jacobian_tracer_profiles():
    model = DummyDecoderModel(num_layers=6, d_model=32)
    corpus = AffectiveCartographyCorpus()
    tracer = AffectiveJacobianTracer(model)

    profiles = tracer.trace_corpus_sensitivity(corpus)
    assert len(profiles) == 6

    for p in profiles:
        assert p.layer_index in range(6)
        assert p.idiom_sensitivity > 0.0
        assert p.crisis_sensitivity > 0.0
        assert p.neutral_sensitivity > 0.0
        assert p.differential_ratio > 0.0


def test_thx02_diagnoser_calibrated_decision():
    model = DummyDecoderModel(num_layers=6, d_model=32)
    corpus = AffectiveCartographyCorpus()
    tracer = AffectiveJacobianTracer(model)
    profiles = tracer.trace_corpus_sensitivity(corpus)

    diagnoser = THX02CircuitDiagnoser()
    diagnosis = diagnoser.diagnose(
        profiles,
        {"idiom_count": 4, "crisis_count": 3},
    )

    assert diagnosis.primary_circuit_locus in ("early_lexical", "mid_pragmatic_affective", "late_unembedding")
    assert diagnosis.attenuation_risk in ("high_crisis_suppression", "cultural_flattening", "well_preserved")
    assert diagnosis.target_layer_index in range(6)
    assert diagnosis.confidence >= 0.70
    assert "mid_pragmatic_affective" in diagnosis.decision_probabilities


def test_subspace_projection_solver_bounded_execution():
    model = DummyDecoderModel(num_layers=6, d_model=32)
    corpus = AffectiveCartographyCorpus()
    tracer = AffectiveJacobianTracer(model)
    profiles = tracer.trace_corpus_sensitivity(corpus)
    diagnoser = THX02CircuitDiagnoser()
    diagnosis = diagnoser.diagnose(profiles, {"idiom_count": 4, "crisis_count": 3})

    solver = SubspaceProjectionSolver(max_allowed_norm=0.05)
    result = solver.solve_bounded_patch(model, diagnosis)

    assert result.status in ("APPLIED", "ABSTAINED_SAFE")
    if result.status == "APPLIED":
        assert result.frobenius_norm_delta > 0.0
        assert result.frobenius_norm_delta <= 0.05
        assert result.clean_projection_preservation_score >= 0.90
        assert result.weights_modified is True
        assert result.diagnostic_receipt["solver"] == "closed_form_rank1_svd_nullspace"


def test_subspace_projection_solver_abstains_on_safe():
    model = DummyDecoderModel(num_layers=6, d_model=32)
    diagnoser = THX02CircuitDiagnoser()

    # Artificially construct safe diagnosis
    from weightliftai.cartography import THX02DiagnosisResult

    safe_diag = THX02DiagnosisResult(
        primary_circuit_locus="mid_pragmatic_affective",
        attenuation_risk="well_preserved",
        target_layer_index=2,
        target_subspace="abstain_safe",
        confidence=0.95,
        decision_probabilities={"mid_pragmatic_affective": 0.95},
        rationale="All markers well preserved.",
        model_identifier="test",
    )

    solver = SubspaceProjectionSolver()
    result = solver.solve_bounded_patch(model, safe_diag)

    assert result.status == "ABSTAINED_SAFE"
    assert result.frobenius_norm_delta == 0.0
    assert result.weights_modified is False


def test_full_affective_cartography_pipeline_end_to_end():
    model = DummyDecoderModel(num_layers=6, d_model=32)
    pipeline = AffectiveCartographyPipeline(model=model)
    report = pipeline.run()

    assert report.num_layers == 6
    assert report.corpus_size >= 8
    assert len(report.layer_profiles) == 6
    assert report.verified_fail_closed is True
    assert report.thx02_diagnosis.confidence > 0.50
