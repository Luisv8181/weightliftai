"""Tests for Jacobian Lens and J-Space Analysis."""

from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from weightliftai.cartography.jacobian_lens import (
    aggregate_jacobian_corpus,
    check_finite_differences,
    compare_logit_lens_and_jacobian_lens,
    compute_vjp,
    get_unembedding_weight,
)


class MockBlock(torch.nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        # Residual connection
        residual = x
        h = self.relu(self.linear1(x))
        out = residual + 0.5 * self.linear2(h)
        return (out,)


class MockLM(torch.nn.Module):
    def __init__(self, n_layers: int = 3, dim: int = 16, vocab_size: int = 32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, dim)
        self.layers = torch.nn.ModuleList([MockBlock(dim) for _ in range(n_layers)])
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


def test_check_finite_differences_matches_autograd():
    torch.manual_seed(42)
    model = MockLM(n_layers=2, dim=8, vocab_size=16)
    input_ids = torch.tensor([[1, 3, 7]])
    target_v = torch.randn(8)
    
    result = check_finite_differences(
        model, input_ids, layer_idx=0, target_vector=target_v,
        epsilon=1e-4, dims_to_check=4
    )
    assert result["passed"] is True
    assert result["max_relative_error"] < 0.05


def test_compare_logit_lens_and_jacobian_lens():
    model = MockLM(n_layers=3, dim=16, vocab_size=32)
    tokenizer = MockTokenizer(vocab_size=32)
    text = "tok_2 tok_5 tok_8"

    result = compare_logit_lens_and_jacobian_lens(
        model, tokenizer, text, layer_idx=1, top_k=5
    )
    assert result["layer"] == 1
    assert len(result["logit_lens_top_k"]) == 5
    assert len(result["jacobian_lens_top_k"]) == 5
    assert -1.0 <= result["cosine_similarity"] <= 1.0
    assert -1.0 <= result["random_baseline_cosine_similarity"] <= 1.0
    assert -1.0 <= result["rank_correlation_spearman"] <= 1.0


def test_aggregate_jacobian_corpus():
    model = MockLM(n_layers=2, dim=8, vocab_size=16)
    tokenizer = MockTokenizer(vocab_size=16)
    corpus = ["tok_1 tok_2", "tok_3 tok_4", "tok_5 tok_6"]

    res = aggregate_jacobian_corpus(
        model, tokenizer, corpus, layer_idx=0, target_token_id=5
    )
    assert res["sample_count"] == 3
    assert len(res["corpus_sha256"]) == 64
    assert res["target_token_id"] == 5
    assert len(res["mean_direction"]) == 8
    assert res["mean_direction_norm"] > 0
    assert res["std_error_mean"] >= 0
