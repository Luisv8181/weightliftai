import pytest

torch = pytest.importorskip("torch")

from weightliftai.activations import capture_activations, summarize_tensor


def test_summarize_tensor_returns_statistics_without_raw_tensor():
    result = summarize_tensor(torch.tensor([[0.0, -1.0, 2.0, 3.0]]), top_k=2)
    assert result["shape"] == [1, 4]
    assert result["numel"] == 4
    assert result["min"] == -1.0
    assert result["max"] == 3.0
    assert result["zero_fraction"] == 0.25
    assert len(result["largest_absolute_values"]) == 2
    assert "values" not in result


def test_capture_selected_module():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )
    result = capture_activations(
        model,
        {"input": torch.ones(1, 3)},
        ["1"],
    )
    assert result["1"]["shape"] == [1, 4]
    assert result["1"]["min"] == 0.0


def test_capture_rejects_unknown_module():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    with pytest.raises(ValueError, match="Unknown module"):
        capture_activations(model, {"input": torch.ones(1, 3)}, ["missing"])
