"""Reproducible Known-Defect Recovery Experiment.

Implements Initial Experiment #1 from docs/WEIGHT_SPACE_PROGRAMMING.md:
1. Synthesizes a ground-truth MLP for multi-feature function approximation.
2. Injects a controlled weight corruption (known defect).
3. Evaluates baseline vs corrupted performance.
4. Synthesizes an exact declarative candidate repair patch.
5. Verifies repair with independent evaluation, distribution-shift tests, and regression gates.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn as nn

from weightliftai.weight_programming.evaluator import IndependentEvaluator
from weightliftai.weight_programming.schemas import PatchBudget, WeightPatch


class ToyMLP(nn.Module):
    """A clean 2-layer MLP for controlled weight-space experiments."""
    def __init__(self, in_features: int = 4, hidden_dim: int = 8, out_features: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def generate_synthetic_data(seed: int = 42) -> dict[str, dict[str, Any]]:
    """Generate in-domain, held-out, distribution-shift, and regression datasets."""
    torch.manual_seed(seed)
    
    # Ground truth mapping: 4 inputs -> 2 outputs (Output 0: Target Task, Output 1: Control/Regression Task)
    true_w = torch.randn(4, 2)
    
    # In-domain
    x_in = torch.randn(64, 4)
    y_in = torch.matmul(x_in, true_w) + 0.05 * torch.randn(64, 2)

    # Held-out test
    x_test = torch.randn(32, 4)
    y_test = torch.matmul(x_test, true_w) + 0.05 * torch.randn(32, 2)

    # Distribution shift (OOD inputs)
    x_ood = 2.0 * torch.randn(32, 4) + 1.0
    y_ood = torch.matmul(x_ood, true_w) + 0.05 * torch.randn(32, 2)

    # Regression suite: Untouched task (Output 1)
    x_reg = torch.randn(32, 4)
    y_reg = torch.matmul(x_reg, true_w) + 0.05 * torch.randn(32, 2)

    return {
        "train_full": {"x": x_in, "y": y_in},
        "in_domain": {"x": x_in, "y": y_in[:, :1], "target_col": 0},
        "held_out": {"x": x_test, "y": y_test[:, :1], "target_col": 0},
        "distribution_shift": {"x": x_ood, "y": y_ood[:, :1], "target_col": 0},
        "regression": {"x": x_reg, "y": y_reg[:, 1:], "target_col": 1},
    }


def mse_evaluator(model: nn.Module, split_data: dict[str, Any]) -> float:
    """Mean squared error evaluation metric (lower is better, returned as negative MSE for maximization)."""
    with torch.no_grad():
        preds = model(split_data["x"])
        col = split_data.get("target_col")
        if col is not None:
            preds = preds[:, col:col+1]
        mse = torch.nn.functional.mse_loss(preds, split_data["y"]).item()
        return -mse


def run_known_defect_experiment() -> dict[str, Any]:
    """Execute the full reproducible defect-injection and repair experiment."""
    data = generate_synthetic_data()

    # 1. Train ground truth model on both outputs
    model = ToyMLP()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(250):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(data["train_full"]["x"]), data["train_full"]["y"])
        loss.backward()
        opt.step()

    # Save clean baseline weights
    clean_fc2_w = model.fc2.weight.data.clone().numpy()

    # 2. Inject controlled defect: zero out connections in Output 0 only (Target Task)
    corrupted_w = clean_fc2_w.copy()
    corrupted_w[0, :4] = 0.0  # Output 1 remains completely intact
    model.fc2.weight.data.copy_(torch.from_numpy(corrupted_w))

    # 3. Create candidate repair patch: restore missing delta in Output 0
    delta_w = clean_fc2_w - corrupted_w
    
    patch = WeightPatch(
        patch_id="patch-defect-repair-001",
        base_model_id="toy-mlp-ground-truth",
        base_model_revision="commit-000",
        deltas={"fc2.weight": delta_w},
        hypothesis="Restoring masked connection weights in Output 0 of fc2 recovers function without regression.",
        proposed_by="diagnostic_specialist",
        rationale="fc2 weights showed truncation in first 4 columns of output 0",
    )

    budget = PatchBudget(
        max_changed_tensors=1,
        max_l2_norm=5.0,
        max_relative_norm=1.0,
        allowed_tensor_names=["fc2.weight"],
    )

    evaluator = IndependentEvaluator(
        eval_fn=mse_evaluator,
        target_in_domain=data["in_domain"],
        target_held_out=data["held_out"],
        distribution_shift=data["distribution_shift"],
        regression_suite=data["regression"],
        min_target_gain=0.10,
        max_regression_tolerance=0.02,
        higher_is_better=True,
    )

    result = evaluator.evaluate_patch(model, patch, budget=budget)
    return result
