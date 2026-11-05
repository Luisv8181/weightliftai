import copy

import torch
from torch import nn

from weightliftai.cartography.representation_intervention import (
    BehavioralFailure,
    CircuitHypothesis,
    EvaluationSample,
    run_activation_ablation_hook,
    run_activation_steering_hook,
    run_traceable_intervention_pipeline,
)


class SimpleDecoderToy(nn.Module):
    """A minimal 2-layer MLP decoder for CPU testing."""
    def __init__(self, d_in: int = 16, d_hidden: int = 32, d_out: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)


def test_activation_steering_and_ablation_hooks():
    torch.manual_seed(42)
    model = SimpleDecoderToy()
    x = torch.randn(2, 16)
    out_orig = model(x)

    steering_vec = torch.randn(32)
    _handle, remove = run_activation_steering_hook(model, "fc1", steering_vec, alpha=2.0)
    out_steered = model(x)
    assert not torch.allclose(out_orig, out_steered)
    remove()

    out_restored = model(x)
    assert torch.allclose(out_orig, out_restored)

    # Test ablation
    _handle_abl, remove_abl = run_activation_ablation_hook(model, "fc1", steering_vec)
    out_ablated = model(x)
    assert not torch.allclose(out_orig, out_ablated)
    remove_abl()
    assert torch.allclose(out_orig, model(x))


def test_traceable_intervention_pipeline_least_invasive_decision():
    torch.manual_seed(42)
    model = SimpleDecoderToy()

    failure = BehavioralFailure(
        task_id="task-001",
        prompt="Sample probe",
        expected_output="class_A",
        observed_output="class_B",
        metric_name="accuracy",
        error_magnitude=1.0,
    )

    hypothesis = CircuitHypothesis(
        hypothesis_id="hyp-001",
        target_module="fc1",
        target_layer=1,
        site="output",
        direction_name="target_steering_dir",
        rationale="Jacobian sensitivity identified fc1 activation dimension as primary gradient source.",
        direction=[1.0] + [0.0] * 31,
    )

    samples = [
        EvaluationSample("s1", "diagnostic sample", "target", "diagnostic"),
        EvaluationSample("s2", "counterfactual sample", "other", "matched_counterfactual"),
        EvaluationSample("s3", "generalization sample", "target", "held_out_generalization"),
        EvaluationSample("s4", "negative control sample", "none", "negative_control"),
    ]

    # Mock evaluate_fn simulating successful steering and safe counterfactuals
    def mock_evaluate(m, eval_samples):
        # Check if steering hook is active by passing probe tensor
        probe = torch.ones(1, 16)
        out = m(probe)
        is_steered = float(out[0, 0].item()) > 0.6

        if is_steered:
            return {
                "diagnostic": {"accuracy": 0.85},
                "matched_counterfactual": {"accuracy": 0.95},
                "held_out_generalization": {"accuracy": 0.80},
                "negative_control": {"accuracy": 0.10},
            }
        # Baseline
        return {
            "diagnostic": {"accuracy": 0.40},
            "matched_counterfactual": {"accuracy": 0.95},
            "held_out_generalization": {"accuracy": 0.50},
            "negative_control": {"accuracy": 0.10},
        }

    def mock_weight_patch(base_model, hyp):
        patched = copy.deepcopy(base_model)
        with torch.no_grad():
            patched.fc1.weight.add_(0.01)
        return patched

    result = run_traceable_intervention_pipeline(
        model=model,
        failure=failure,
        hypothesis=hypothesis,
        eval_samples=samples,
        evaluate_fn=mock_evaluate,
        weight_patch_fn=mock_weight_patch,
        steering_alpha=2.5,
    )

    assert result.pipeline_id == "pipe-hyp-001"
    assert result.diagnostic_recovery["baseline"] == 0.40
    assert result.diagnostic_recovery["activation_steering"] == 0.85
    assert result.least_invasive_recommendation["chosen_intervention"] == "activation_steering"
    assert result.least_invasive_recommendation["activation_sufficient"] is True
    assert result.least_invasive_recommendation["weight_edit_justified"] is False
    assert len(result.limitations) > 0
