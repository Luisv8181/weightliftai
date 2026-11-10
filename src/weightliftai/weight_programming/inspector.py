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
    BaseModelMismatchError,
    BudgetViolationError,
    DtypeMismatchError,
    PatchBudget,
    WeightPatch,
)


def hash_tensor_bytes(param: nn.Parameter) -> str:
    """Hash exact native parameter tensor bytes, preserving dtype and shape without float32 casting."""
    hasher = hashlib.sha256()
    hasher.update(str(param.dtype).encode("utf-8"))
    hasher.update(np.array(param.shape, dtype=np.int64).tobytes())
    t = param.detach().cpu().contiguous()
    hasher.update(t.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def compute_model_fingerprint(model: nn.Module) -> str:
    """Compute deterministic cryptographic root SHA256 of all model parameters."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(hash_tensor_bytes(param).encode("utf-8"))
    return hasher.hexdigest()


def inspect_model_parameters(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Return an exact inspection dictionary for all model parameters with native byte hashes."""
    report = {}
    for name, param in model.named_parameters():
        arr = param.detach().cpu().float().numpy()
        l2 = float(np.linalg.norm(arr))
        param_hash = hash_tensor_bytes(param)
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
    require_fingerprint: bool = False,
) -> dict[str, Any]:
    """Validate candidate patch against budget constraints, allowlists, and base checkpoint fingerprint.
    
    Raises:
    - BaseModelMismatchError if patch declared a base fingerprint that does not match the model or missing when required.
    - AllowlistViolationError if an unpermitted tensor is targeted or tied-parameter conflict exists.
    - BudgetViolationError / DtypeMismatchError if tensor count, dtype, non-finite values (NaN/Inf), L2 norm, or relative norm limits are exceeded.
    """
    # 1. Base model fingerprint verification
    if require_fingerprint and patch.base_model_fingerprint is None:
        raise BaseModelMismatchError(
            "Candidate patch lacks required base_model_fingerprint. Base fingerprint verification cannot be bypassed."
        )
    if patch.base_model_fingerprint is not None:
        actual_fingerprint = compute_model_fingerprint(model)
        if actual_fingerprint != patch.base_model_fingerprint:
            raise BaseModelMismatchError(
                f"Base model fingerprint mismatch: expected {patch.base_model_fingerprint}, got {actual_fingerprint}"
            )

    # 2. Reject empty patch
    if len(patch.deltas) == 0:
        raise BudgetViolationError("Patch contains no parameter deltas.")

    # 3. Check tensor count budget
    if len(patch.deltas) > budget.max_changed_tensors:
        raise BudgetViolationError(
            f"Patch modifies {len(patch.deltas)} tensors; budget limit is {budget.max_changed_tensors}"
        )

    # 4. Check allowlist if configured
    if budget.allowed_tensor_names is not None:
        allowed_set = set(budget.allowed_tensor_names)
        unauthorized = set(patch.deltas.keys()) - allowed_set
        if unauthorized:
            raise AllowlistViolationError(
                f"Patch targets tensors not on allowlist: {sorted(unauthorized)}"
            )

    # 5. Check tied-weight / memory-aliasing conflicts across all parameter paths
    tensor_ptr_map: dict[int, str] = {}
    for p_name, p in model.named_parameters(remove_duplicate=False):
        ptr = p.data_ptr()
        if ptr in tensor_ptr_map:
            other = tensor_ptr_map[ptr]
            if (p_name in patch.deltas) != (other in patch.deltas):
                raise AllowlistViolationError(
                    f"Tied-parameter conflict: '{p_name}' shares storage with '{other}', but only one is declared in patch."
                )
        else:
            tensor_ptr_map[ptr] = p_name

    model_params = dict(model.named_parameters(remove_duplicate=False))
    total_delta_l2_sq = 0.0
    validation_details = {}

    for name, delta in patch.deltas.items():
        if name not in model_params:
            raise ValueError(f"Target tensor '{name}' does not exist in model.")
        
        # 6. Reject non-floating or incompatible dtypes
        if not np.issubdtype(delta.dtype, np.floating):
            raise DtypeMismatchError(
                f"Tensor '{name}' delta has incompatible non-floating dtype '{delta.dtype}'. Floating-point delta required."
            )

        # 7. Reject non-finite values (NaN / Inf)
        if not np.isfinite(delta).all():
            raise BudgetViolationError(
                f"Tensor '{name}' delta contains non-finite values (NaN or Inf)."
            )

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
