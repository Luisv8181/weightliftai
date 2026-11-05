import pytest

torch = pytest.importorskip("torch")

from weightliftai.activations import (
    capture_activations,
    compare_activations,
    compare_tensors,
    summarize_tensor,
)


def test_summarize_tensor_returns_statistics_without_raw_tensor():
    result = summarize_tensor(torch.tensor([[0.0, -1.0, 2.0, 3.0]]), top_k=2)
    assert result["shape"] == [1, 4]
    assert result["numel"] == 4
    assert result["min"] == -1.0
    assert result["max"] == 3.0
    assert result["zero_fraction"] == 0.25
    assert len(result["largest_absolute_values"]) == 2
    assert "values" not in result


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
        )

    def forward(self, input):
        return self.layers(input)


def test_capture_selected_module():
    torch.manual_seed(0)
    model = TinyModel()
    result = capture_activations(model, {"input": torch.ones(1, 3)}, ["layers.1"])
    assert result["layers.1"]["shape"] == [1, 4]
    assert result["layers.1"]["min"] == 0.0


def test_capture_rejects_unknown_module():
    model = TinyModel()
    with pytest.raises(ValueError, match="Unknown module"):
        capture_activations(model, {"input": torch.ones(1, 3)}, ["missing"])


def test_compare_tensors_identical_vectors():
    t = torch.tensor([[1.0, 2.0, 3.0]])
    cmp = compare_tensors(t, t, top_k=2)
    assert pytest.approx(cmp["cosine_similarity"], abs=1e-5) == 1.0
    assert pytest.approx(cmp["l2_distance"], abs=1e-5) == 0.0
    assert pytest.approx(cmp["norm_ratio"], abs=1e-5) == 1.0
    assert len(cmp["top_divergent_dimensions"]) == 2
    for item in cmp["top_divergent_dimensions"]:
        assert item["absolute_difference"] == 0.0


def test_compare_tensors_orthogonal_vectors():
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 5.0, 0.0]])
    cmp = compare_tensors(a, b, top_k=1)
    assert pytest.approx(cmp["cosine_similarity"], abs=1e-5) == 0.0
    assert pytest.approx(cmp["norm_ratio"], abs=1e-5) == 0.2
    assert cmp["top_divergent_dimensions"][0]["dimension"] == 1
    assert pytest.approx(cmp["top_divergent_dimensions"][0]["absolute_difference"], abs=1e-5) == 5.0


def test_compare_tensors_pooling_options():
    a = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    b = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]]])
    cmp_mean = compare_tensors(a, b, pooling="mean")
    assert cmp_mean["cosine_similarity"] > 0.99
    cmp_last = compare_tensors(a, b, pooling="last")
    assert cmp_last["top_divergent_dimensions"][0]["dimension"] == 1
    assert pytest.approx(cmp_last["top_divergent_dimensions"][0]["absolute_difference"], abs=1e-5) == 2.0


def test_compare_activations_tiny_model():
    model = TinyModel()
    inp_a = {"input": torch.ones(1, 3)}
    inp_b = {"input": -torch.ones(1, 3)}
    results = compare_activations(model, inp_a, inp_b, ["layers.0", "layers.1"], pooling="mean")
    assert "layers.0" in results
    assert "layers.1" in results
    assert "cosine_similarity" in results["layers.0"]
    assert "top_divergent_dimensions" in results["layers.1"]
