"""Read-only model parameter inspector and patch budget validator.

Provides deterministic diagnostics of model parameters and enforces safety guardrails
before any candidate intervention can be executed.
"""

from __future__ import annotations

import hashlib
from typing import Any
import numpy as np
import torch
import torch.nn as nn

from weightliftai.weight_programming.schemas import (
    AllowlistViolationError,
    BudgetViolationError,
    PatchBudget,
    WeightPatch,
)


def inspect_model_parameters(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Return an exact inspection dictionary for all model parameters."""
    report = {}
    for name, param in model.named_parameters():
        arr = param.detach().cpu().float().numpy()
        l2 = float(np.linalg.norm(arr))
        param_hash = hashlib.sha256(arr.tobytes()).hexdigest()
        report[name] = {
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "numel": param.numel(),
            "l2_norm": l2,
            "requires_grad": param.requires_grad,
            "sha256": param_hash,
        }
    return report


def validate_patch_against_budget(
    model: nn.Module,
    patch: WeightPatch,
    budget: PatchBudget,
) -> dict[str, Any]:
    """Validate candidate patch against budget constraints and tensor allowlists.
    
    Raises:
    - AllowlistViolationError if an unpermitted tensor is targeted.
    - BudgetViolationError if tensor count, L2 norm, or relative norm limits are exceeded.
    """
    model_params = dict(model.named_parameters())
    
    # Check tensor count budget
    if len(patch.deltas) > budget.max_changed_tensors:
        raise BudgetViolationError(
            f"Patch modifies {len(patch.deltas)} tensors; budget limit is {budget.max_changed_tensors}"
        )

    # Check allowlist if configured
    if budget.allowed_tensor_names is not None:
        allowed_set = set(budget.allowed_tensor_names)
        unauthorized = set(patch.deltas.keys()) - allowed_set
        if unauthorized:
            raise AllowlistViolationError(
                f"Patch targets tensors not on allowlist: {sorted(unauthorized)}"
            )

    total_delta_l2_sq = 0.0
    validation_details = {}

    for name, delta in patch.deltas.items():
        if name not in model_params:
            raise ValueError(f"Target tensor '{name}' does not exist in model.")
        
        orig_p = model_params[name].detach().cpu().float().numpy()
        if orig_p.shape != delta.shape:
            raise ValueError(
                f"Delta shape {delta.shape} does not match parameter shape {orig_p.shape} for '{name}'"
            )

        delta_l2 = float(np.linalg.norm(delta))
        orig_l2 = float(np.linalg.norm(orig_p))
        rel_norm = delta_l2 / max(orig_l2, 1e-8)

        total_delta_l2_sq += delta_l2 ** 2

        if rel_norm > budget.max_relative_norm:
            raise BudgetViolationError(
                f"Tensor '{name}' relative delta {rel_norm:.4f} exceeds budget limit {budget.max_relative_norm:.4f}"
            )

        validation_details[name] = {
            "delta_l2": delta_l2,
            "orig_l2": orig_l2,
            "relative_norm": rel_norm,
        }

    total_l2 = float(np.sqrt(total_delta_l2_sq))
    if total_l2 > budget.max_l2_norm:
        raise BudgetViolationError(
            f"Total patch L2 norm {total_l2:.4f} exceeds budget limit {budget.max_l2_norm:.4f}"
        )

    return {
        "status": "passed",
        "total_delta_l2": total_l2,
        "max_l2_allowed": budget.max_l2_norm,
        "tensors_validated": len(patch.deltas),
        "details": validation_details,
    }
