"""Tests for Jacobian Lens, Multi-Context/Cross-Position Aggregation, and 4-Dimension Evidence Separation (Prompts 28, 29, 31)."""

from __future__ import annotations

from pathlib import Path
import tomllib
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import weightliftai.cartography as cartography_pkg
from weightliftai.cartography.jacobian_lens import (
    JACOBIAN_LENS_METHOD_MAPPING,
    aggregate_jacobian_corpus,
    check_finite_differences,
    compare_logit_lens_and_jacobian_lens,
    compute_vjp,
    evaluate_four_dimension_jacobian_evidence,
    get_unembedding_weight,
)


class MockCausalBlock(torch.nn.Module):
    """Mock transformer block with causal cross-token mixing so token t influences future tokens t' >= t."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.linear_self = torch.nn.Linear(dim, dim)
        self.linear_prev = torch.nn.Linear(dim, dim, bias=False)
        self.relu = torch.nn.ReLU()
        self.linear_out = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        residual = x
        # Causal prefix cumulative mean mixes source positions t into target positions t' >= t
        seq_len = x.shape[1]
        tril = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=x.dtype))
        denom = tril.sum(dim=-1, keepdim=True).clamp_min(1.0)
        causal_ctx = torch.matmul(tril / denom, x)
        h = self.relu(self.linear_self(x) + 0.4 * self.linear_prev(causal_ctx))
        out = residual + 0.5 * self.linear_out(h)
        return (out,)


class MockLM(torch.nn.Module):
    def __init__(self, n_layers: int = 3, dim: int = 16, vocab_size: int = 32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, dim)
        self.layers = torch.nn.ModuleList([MockCausalBlock(dim) for _ in range(n_layers)])
        self.lm_head = torch.nn.Linear(dim, vocab_size, bias=False)
        self.config = SimpleNamespace(num_hidden_layers=n_layers)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        logits = self.lm_head(x)
        return SimpleNamespace(last_hidden_state=x, logits=logits)


class MockTokenizer:
    def __init__(self, vocab_size: int = 32):
        self.vocab = {i: f"tok_{i}" for i in range(vocab_size)}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __call__(self, text, return_tensors="pt", **_kwargs):
        words = text.split()
        ids = [self.inv_vocab.get(w, 1) for w in words]
        if not ids:
            ids = [0]
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor}

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return " ".join(self.vocab.get(t, f"tok_{t}") for t in token_ids)


def test_cartography_public_imports_and_pyproject_dependencies():
    """Verify public cartography exports and pyproject.toml dependency completeness (Prompt 29)."""
    for sym in (
        "JACOBIAN_LENS_METHOD_MAPPING",
        "compute_vjp",
        "check_finite_differences",
        "compare_logit_lens_and_jacobian_lens",
        "aggregate_jacobian_corpus",
        "evaluate_four_dimension_jacobian_evidence",
        "project_representations",
        "compute_distortion_metrics",
    ):
        assert hasattr(cartography_pkg, sym), f"Missing public symbol {sym} on weightliftai.cartography"

    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    cart_deps = " ".join(data["project"]["optional-dependencies"]["cartography"])
    for req_pkg in ("numpy", "torch", "pandas", "scikit-learn", "scipy", "pydantic", "umap-learn"):
        assert req_pkg in cart_deps, f"Package '{req_pkg}' missing from [project.optional-dependencies.cartography]"


def test_get_unembedding_weight():
    model = MockLM(n_layers=2, dim=8, vocab_size=10)
    w = get_unembedding_weight(model)
    assert w.shape == (10, 8)


def test_compute_vjp_shape_and_values():
    model = MockLM(n_layers=3, dim=16, vocab_size=32)
    input_ids = torch.tensor([[2, 5, 8]])
    target_v = torch.randn(16)

    vjp = compute_vjp(model, input_ids, layer_idx=1, target_vector=target_v, position=-1)
    assert vjp.shape == (16,)
    assert not torch.isnan(vjp).any()
    assert torch.norm(vjp) > 0


def test_cross_position_vjp_and_causal_horizon_enforcement():
    """Verify distinct source t and target t' >= t positions and causal horizon enforcement (Prompt 28)."""
    torch.manual_seed(42)
    model = MockLM(n_layers=3, dim=12, vocab_size=24)
    input_ids = torch.tensor([[2, 5, 8, 11]])
    target_v = torch.randn(12)

    # Cross-position VJP: source t=0 influencing future target t'=3
    vjp_cross = compute_vjp(
        model,
        input_ids,
        layer_idx=0,
        target_vector=target_v,
        source_position=0,
        target_position=3,
    )
    # Same-position VJP: source t=3 influencing target t'=3
    vjp_same = compute_vjp(
        model,
        input_ids,
        layer_idx=0,
        target_vector=target_v,
        source_position=3,
        target_position=3,
    )
    assert vjp_cross.shape == (12,)
    assert torch.norm(vjp_cross) > 0
    # Regression check: cross-position (t=0 -> t'=3) is distinct from same-position (t=3 -> t'=3)
    assert not torch.allclose(vjp_cross, vjp_same)

    # Causal check: target_position < source_position is rejected
    with pytest.raises(ValueError, match="Causal horizon violation"):
        compute_vjp(
            model,
            input_ids,
            layer_idx=0,
            target_vector=target_v,
            source_position=2,
            target_position=1,
        )


def test_check_finite_differences_matches_autograd_same_and_cross_position():
    torch.manual_seed(42)
    model = MockLM(n_layers=2, dim=8, vocab_size=16)
    input_ids = torch.tensor([[1, 3, 7, 9]])
    target_v = torch.randn(8)

    # Same position
    result_same = check_finite_differences(
        model,
        input_ids,
        layer_idx=0,
        target_vector=target_v,
        epsilon=1e-4,
        dims_to_check=4,
    )
    assert result_same["passed"] is True
    assert result_same["max_relative_error"] < 0.05

    # Cross-position (source t=1 -> future target t'=3)
    result_cross = check_finite_differences(
        model,
        input_ids,
        layer_idx=0,
        target_vector=target_v,
        source_position=1,
        target_position=3,
        epsilon=1e-4,
        dims_to_check=4,
    )
    assert result_cross["passed"] is True
    assert result_cross["source_position"] == 1
    assert result_cross["target_positions"] == [3]
    assert result_cross["max_relative_error"] < 0.05


def test_compare_logit_lens_and_jacobian_lens():
    model = MockLM(n_layers=3, dim=16, vocab_size=32)
    tokenizer = MockTokenizer(vocab_size=32)
    text = "tok_2 tok_5 tok_8"

    result = compare_logit_lens_and_jacobian_lens(
        model, tokenizer, text, layer_idx=1, top_k=5, source_position=0, target_position=2
    )
    assert result["layer"] == 1
    assert result["source_position"] == 0
    assert result["target_positions"] == [2]
    assert len(result["logit_lens_top_k"]) == 5
    assert len(result["jacobian_lens_top_k"]) == 5
    assert result["candidate_pool_size"] >= 5
    assert result["shortlist_mode"] == "union_logit_and_final_prediction"
    assert -1.0 <= result["cosine_similarity"] <= 1.0
    assert -1.0 <= result["norm_matched_random_cosine_similarity"] <= 1.0
    assert -1.0 <= result["rank_correlation_spearman"] <= 1.0


def test_aggregate_jacobian_corpus_multi_context_and_future_positions():
    torch.manual_seed(42)
    model = MockLM(n_layers=2, dim=8, vocab_size=16)
    tokenizer = MockTokenizer(vocab_size=16)
    corpus = [
        "tok_1 tok_2 tok_3",
        "tok_4 tok_5 tok_6",
        "tok_7 tok_8 tok_9",
    ]

    res = aggregate_jacobian_corpus(
        model,
        tokenizer,
        corpus,
        layer_idx=0,
        target_token_id=5,
        source_positions="all",
        future_offsets=[0, 1],
        aggregation_mode="trimmed_mean",
        normalization_mode="l2",
    )
    assert res["sample_count"] == 3
    # Each 3-token context has (0->0, 0->1, 1->1, 1->2, 2->2) = 5 pairs -> 15 total pairs
    assert res["evaluated_pair_count"] == 15
    assert len(res["corpus_sha256"]) == 64
    assert res["target_token_id"] == 5
    assert len(res["mean_direction"]) == 8
    assert res["mean_direction_norm"] > 0
    assert res["std_error_mean"] >= 0
    assert -1.0 <= res["cross_context_cosine_stability"] <= 1.0


def test_four_dimension_jacobian_evidence_separation():
    """Verify separate reporting of Cosine Alignment, Sensitivity, Stability, and Intervention Effects (Prompt 31)."""
    torch.manual_seed(42)
    model = MockLM(n_layers=2, dim=8, vocab_size=16)
    tokenizer = MockTokenizer(vocab_size=16)
    corpus = [
        "tok_1 tok_2 tok_3",
        "tok_4 tok_5 tok_6",
        "tok_7 tok_8 tok_9",
    ]

    report = evaluate_four_dimension_jacobian_evidence(
        model,
        tokenizer,
        corpus,
        layer_idx=0,
        target_token_id=5,
        control_token_id=2,
        source_positions="all",
        future_offsets=(0, 1),
        intervention_alpha=0.5,
    )

    assert report["method_mapping"] == JACOBIAN_LENS_METHOD_MAPPING
    d1 = report["dimension_1_cosine_alignment"]
    d2 = report["dimension_2_sensitivity"]
    d3 = report["dimension_3_stability"]
    d4 = report["dimension_4_intervention_effects"]

    assert "observed_cosine_alignment" in d1
    assert "norm_matched_random_control_cosine" in d1
    assert "unrelated_token_control_cosine" in d1
    assert "non_claim_guardrail" in d1

    assert d2["mean_vjp_l2_norm"] > 0.0
    assert "std_vjp_l2_norm" in d2

    assert d3["evaluated_context_count"] == 3
    assert d3["evaluated_pair_count"] == 15
    assert -1.0 <= d3["cross_context_pairwise_cosine_mean"] <= 1.0

    assert "candidate_mean_target_logit_shift" in d4
    assert "random_control_mean_target_logit_shift" in d4
    assert "orthogonal_sham_mean_target_logit_shift" in d4
    assert isinstance(d4["causal_claim_supported"], bool)
