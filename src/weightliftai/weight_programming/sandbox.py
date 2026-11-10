"""Sandboxed parameter execution with bit-exact reversible rollback.

Base checkpoints are strictly immutable. Patches are either evaluated on disposable
clones or inside a rollback context manager that verifies parameter checksums before
and after execution.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
from typing import Generator
import numpy as np
import torch
import torch.nn as nn

from weightliftai.weight_programming.inspector import hash_tensor_bytes, validate_patch_against_budget
from weightliftai.weight_programming.schemas import BudgetViolationError, PatchBudget, WeightPatch


def _hash_tensor(param: nn.Parameter) -> str:
    return hash_tensor_bytes(param)


@contextmanager
def apply_patch_sandboxed(
    model: nn.Module,
    patch: WeightPatch,
    budget: PatchBudget | None = None,
    require_budget: bool = True,
    require_fingerprint: bool = False,
) -> Generator[nn.Module, None, None]:
    """Safely apply parameter deltas in memory with verified bit-exact rollback on exit.
    
    Guarantees:
    - Base parameters are validated against a required non-null PatchBudget prior to mutation.
    - Rejects budget omission when require_budget=True.
    - Rejects missing base checkpoint fingerprints when require_fingerprint=True.
    - Rejects non-finite values (NaN/Inf), dtypes, and shape mismatches before parameter mutation.
    - Original tensors are backed up before modification.
    - Exiting the context block restores exact original tensor bytes and validates checksums.
    """
    if budget is None and require_budget:
        raise BudgetViolationError(
            "A non-null PatchBudget is required for sandboxed patch application. "
            "Patch safety cannot be bypassed by omission."
        )

    if budget is not None:
        validate_patch_against_budget(
            model,
            patch,
            budget,
            require_fingerprint=require_fingerprint,
        )

    model_params = dict(model.named_parameters())
    backups: dict[str, torch.Tensor] = {}
    original_hashes: dict[str, str] = {}

    for name in patch.deltas:
        param = model_params[name]
        backups[name] = param.data.clone()
        original_hashes[name] = _hash_tensor(param)

    # Apply deltas
    try:
        with torch.no_grad():
            for name, delta in patch.deltas.items():
                delta_t = torch.from_numpy(delta).to(
                    device=model_params[name].device,
                    dtype=model_params[name].dtype,
                )
                model_params[name].add_(delta_t)
        yield model
    finally:
        # Revert to original tensors
        with torch.no_grad():
            for name, orig_tensor in backups.items():
                model_params[name].copy_(orig_tensor)
                restored_hash = _hash_tensor(model_params[name])
                if restored_hash != original_hashes[name]:
                    raise RuntimeError(
                        f"Rollback failed for '{name}': hash mismatch ({restored_hash} != {original_hashes[name]})"
                    )


def apply_patch_to_copy(
    model: nn.Module,
    patch: WeightPatch,
    budget: PatchBudget | None = None,
    require_budget: bool = True,
    require_fingerprint: bool = False,
) -> nn.Module:
    """Create a deep copy of the model and apply the patch to the clone.
    
    Fails closed if budget is missing when require_budget=True or if budget is violated.
    """
    if budget is None and require_budget:
        raise BudgetViolationError(
            "A non-null PatchBudget is required for clone patch application. "
            "Patch safety cannot be bypassed by omission."
        )

    if budget is not None:
        validate_patch_against_budget(
            model,
            patch,
            budget,
            require_fingerprint=require_fingerprint,
        )

    cloned_model = copy.deepcopy(model)
    cloned_params = dict(cloned_model.named_parameters())

    with torch.no_grad():
        for name, delta in patch.deltas.items():
            delta_t = torch.from_numpy(delta).to(
                device=cloned_params[name].device,
                dtype=cloned_params[name].dtype,
            )
            cloned_params[name].add_(delta_t)

    return cloned_model
