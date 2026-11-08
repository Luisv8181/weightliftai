"""Tests for Layer Atlas representation extraction, paired metrics, and hook cleanup."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from weightliftai.cartography.layer_atlas import (
    compute_layer_metrics,
    content_token_mask,
    extract_layer_representations,
    generate_atlas_html,
)


class MockDecoderLayer(torch.nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return (self.linear(x),)


class MockTransformerModel(torch.nn.Module):
    def __init__(self, n_layers: int = 4, dim: int = 16):
        super().__init__()
        self.layers = torch.nn.ModuleList([MockDecoderLayer(dim) for _ in range(n_layers)])
        self.config = type("Config", (), {"num_hidden_layers": n_layers})()

    def forward(self, input_ids, attention_mask=None):
        # Create dummy embeddings from input_ids [batch, seq] -> [batch, seq, dim]
        x = input_ids.float().unsqueeze(-1).repeat(1, 1, 16)
        for layer in self.layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        return x


class MockTokenizer:
    def __call__(self, texts, return_tensors="pt", padding=True):
        max_len = max(len(t.split()) for t in texts) + 2
        batch_ids = []
        batch_masks = []
        for t in texts:
            words = t.split()
            ids = [1] + [hash(w) % 1000 + 10 for w in words]  # 1 is BOS
            mask = [1] * len(ids)
            # Pad
            pad_len = max_len - len(ids)
            ids += [0] * pad_len
            mask += [0] * pad_len
            batch_ids.append(ids)
            batch_masks.append(mask)
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
        }


def test_extract_layer_representations_shape_and_cleanup():
    model = MockTransformerModel(n_layers=4, dim=16)
    tok = MockTokenizer()
    texts = ["I feel very down today.", "Short sentence.", "A much longer sentence to test padding handling."]
    
    # Extract layers 0 and 2
    acts = extract_layer_representations(
        model=model,
        tokenizer=tok,
        texts=texts,
        layers=[0, 2],
        token_mask="content",
        pooling="mean",
        batch_size=2,
    )

    assert 0 in acts and 2 in acts
    assert acts[0].shape == (3, 16)
    assert acts[2].shape == (3, 16)
    assert np.isfinite(acts[0]).all()

    # Verify all hooks were cleaned up
    for layer in model.layers:
        assert len(layer._forward_hooks) == 0


def test_extract_layer_representations_hook_cleanup_on_error():
    model = MockTransformerModel(n_layers=4, dim=16)
    tok = MockTokenizer()
    
    # Force an error inside forward
    def broken_call(*args, **kwargs):
        raise RuntimeError("Simulated forward pass failure")

    orig_forward = model.forward
    model.forward = broken_call

    with pytest.raises(RuntimeError, match="Simulated forward pass failure"):
        extract_layer_representations(
            model=model,
            tokenizer=tok,
            texts=["Test sentence."],
            layers=[1, 3],
        )

    # All hooks must be cleaned up even after the exception
    for layer in model.layers:
        assert len(layer._forward_hooks) == 0

    model.forward = orig_forward


def test_compute_layer_metrics():
    # 4 items with 3 variants each = 12 rows
    data = []
    rng = np.random.default_rng(42)
    embeddings = rng.normal(0, 1, size=(12, 16))

    variants = ["canonical", "lay", "idiom"]
    for item_idx in range(4):
        for v in variants:
            data.append({
                "item_id": f"item_{item_idx}",
                "variant": v,
                "concept": f"concept_{item_idx % 2}",
                "text": f"text for item {item_idx} variant {v}",
                "lang": "en" if v != "idiom" else "es",
            })
    df = pd.DataFrame(data)

    metrics = compute_layer_metrics(
        embeddings=embeddings,
        df=df,
        item_id_col="item_id",
        variant_col="variant",
        variants=variants,
        n_bootstrap=100,
        seed=42,
    )

    assert "pairwise_cosines" in metrics
    assert metrics["evidence_type"] == "dimension_1_cosine_alignment"
    assert "observational directional similarity" in metrics["guardrail"].lower()
    assert "canonical~lay" in metrics["pairwise_cosines"]
    assert "mean" in metrics["pairwise_cosines"]["canonical~lay"]
    assert "ci_95" in metrics["pairwise_cosines"]["canonical~lay"]
    assert len(metrics["pairwise_cosines"]["canonical~lay"]["ci_95"]) == 2
    assert "unrelated_negative_control" in metrics
    assert "mean" in metrics["unrelated_negative_control"]
    assert "delta_over_unrelated_baseline" in metrics["pairwise_cosines"]["canonical~lay"]


def test_content_token_mask_left_right_padding_and_single_token():
    # Right padding: BOS, tok1, tok2, pad, pad
    attn_right = torch.tensor([[1, 1, 1, 0, 0]])
    m_right = content_token_mask(attn_right)
    # BOS (index 0) zeroed, tok1 and tok2 (index 1, 2) kept
    assert m_right.tolist() == [[0, 1, 1, 0, 0]]

    # Left padding: pad, pad, BOS, tok1, tok2
    attn_left = torch.tensor([[0, 0, 1, 1, 1]])
    m_left = content_token_mask(attn_left)
    # BOS (index 2) zeroed, tok1 and tok2 (index 3, 4) kept
    assert m_left.tolist() == [[0, 0, 0, 1, 1]]

    # Single token sequence: pad, tok1, pad
    attn_single = torch.tensor([[0, 1, 0]])
    m_single = content_token_mask(attn_single)
    # Must preserve single token rather than zeroing entire sequence
    assert m_single.tolist() == [[0, 1, 0]]


def test_extract_layer_representations_last_token_padding_handling():
    model = MockTransformerModel(n_layers=2, dim=16)
    tok = MockTokenizer()
    # One short text (2 words + BOS = 3 tokens), one longer text (5 words + BOS = 6 tokens)
    # The short text will have 3 padding tokens on the right
    texts = ["short one", "this is a longer test sentence"]

    acts_last_pool = extract_layer_representations(
        model=model,
        tokenizer=tok,
        texts=texts,
        layers=[0],
        pooling="last",
        batch_size=2,
    )
    assert acts_last_pool[0].shape == (2, 16)
    assert np.isfinite(acts_last_pool[0]).all()

    acts_last_mask = extract_layer_representations(
        model=model,
        tokenizer=tok,
        texts=texts,
        layers=[0],
        token_mask="last",
        pooling="mean",
        batch_size=2,
    )
    assert acts_last_mask[0].shape == (2, 16)
    # Pooling="last" and token_mask="last" with pooling="mean" must yield identical vectors
    np.testing.assert_allclose(acts_last_pool[0], acts_last_mask[0], rtol=1e-5)


def test_generate_atlas_html():
    layer_metrics = {
        0: {
            "depth_pct": 0.0,
            "pairwise_cosines": {
                "canonical~lay": {"mean": 0.85, "ci_95": [0.82, 0.88]},
                "canonical~idiom": {"mean": 0.70, "ci_95": [0.65, 0.75]},
            },
        },
        13: {
            "depth_pct": 50.0,
            "pairwise_cosines": {
                "canonical~lay": {"mean": 0.92, "ci_95": [0.89, 0.95]},
                "canonical~idiom": {"mean": 0.88, "ci_95": [0.84, 0.91]},
            },
        },
    }
    html_out = generate_atlas_html(
        model_name="TestModel",
        dataset_name="TestDataset",
        layer_metrics=layer_metrics,
    )
    assert "<!doctype html>" in html_out
    assert "Layer Atlas" in html_out
    assert "canonical~lay" in html_out
    assert "canonical~idiom" in html_out
