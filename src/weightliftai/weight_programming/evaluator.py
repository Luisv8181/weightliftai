"""Independent evaluation harness for candidate parameter patches.

Enforces frozen evaluation suites, distribution-shift checks, and regression gates.
The optimizer cannot alter evaluation datasets or redefine acceptance criteria.
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np
import torch
import torch.nn as nn

from weightliftai.weight_programming.sandbox import apply_patch_sandboxed
from weightliftai.weight_programming.schemas import PatchBudget, WeightPatch


class IndependentEvaluator:
    """Evaluates candidate patches against frozen target, held-out, OOD, and regression suites."""

    def __init__(
        self,
        *,
        eval_fn: Callable[[nn.Module, dict[str, Any]], float],
        target_in_domain: dict[str, Any],
        target_held_out: dict[str, Any],
        distribution_shift: dict[str, Any],
        regression_suite: dict[str, Any],
        min_target_gain: float = 0.05,
        max_regression_tolerance: float = 0.02,
        higher_is_better: bool = True,
    ):
        self.eval_fn = eval_fn
        self.target_in_domain = target_in_domain
        self.target_held_out = target_held_out
        self.distribution_shift = distribution_shift
        self.regression_suite = regression_suite
        self.min_target_gain = min_target_gain
        self.max_regression_tolerance = max_regression_tolerance
        self.higher_is_better = higher_is_better

    def evaluate_model(self, model: nn.Module) -> dict[str, float]:
        """Compute metrics across all frozen splits for a given model state."""
        return {
            "in_domain": float(self.eval_fn(model, self.target_in_domain)),
            "held_out": float(self.eval_fn(model, self.target_held_out)),
            "distribution_shift": float(self.eval_fn(model, self.distribution_shift)),
            "regression_suite": float(self.eval_fn(model, self.regression_suite)),
        }

    def evaluate_patch(
        self,
        base_model: nn.Module,
        patch: WeightPatch,
        budget: PatchBudget | None = None,
    ) -> dict[str, Any]:
        """Evaluate baseline and patched model under sandboxed isolation.
        
        Verifies target improvement without regression on untouched capabilities.
        """
        # Baseline evaluation
        base_metrics = self.evaluate_model(base_model)

        # Patched evaluation inside verified sandbox
        with apply_patch_sandboxed(base_model, patch, budget=budget) as patched_model:
            patched_metrics = self.evaluate_model(patched_model)

        # Compute gains/deltas
        sign = 1.0 if self.higher_is_better else -1.0
        target_gain = sign * (patched_metrics["held_out"] - base_metrics["held_out"])
        ood_gain = sign * (patched_metrics["distribution_shift"] - base_metrics["distribution_shift"])
        
        # Regression is measured as drop on untouched regression suite
        regression_drop = sign * (base_metrics["regression_suite"] - patched_metrics["regression_suite"])

        passed_target = bool(target_gain >= self.min_target_gain)
        passed_regression = bool(regression_drop <= self.max_regression_tolerance)
        promotable = passed_target and passed_regression

        return {
            "patch_id": patch.patch_id,
            "patch_sha256": patch.sha256(),
            "passed_gates": promotable,
            "recommendation": "promote" if promotable else "reject",
            "metrics": {
                "baseline": base_metrics,
                "patched": patched_metrics,
            },
            "gains": {
                "target_held_out_gain": float(target_gain),
                "distribution_shift_gain": float(ood_gain),
                "regression_drop": float(max(0.0, regression_drop)),
            },
            "gate_thresholds": {
                "min_target_gain": self.min_target_gain,
                "max_regression_tolerance": self.max_regression_tolerance,
            },
        }
