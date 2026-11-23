"""Tests for Phase 2: Fail-closed patch safety, strict budget enforcement,
dtype checks, fingerprint binding, and negative failure modes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from weightliftai.weight_programming import (
    AllowlistViolationError,
    BaseModelMismatchError,
    BudgetViolationError,
    DtypeMismatchError,
    PatchBudget,
    ToyMLP,
    WeightPatch,
    apply_patch_sandboxed,
    apply_patch_to_copy,
    compute_model_fingerprint,
    validate_patch_against_budget,
)


@pytest.fixture
def clean_model() -> ToyMLP:
    return ToyMLP(in_features=4, hidden_dim=6, out_features=2)


@pytest.fixture
def standard_patch(clean_model: ToyMLP) -> WeightPatch:
    fp = compute_model_fingerprint(clean_model)
    delta = np.ones((6, 4), dtype=np.float32) * 0.05
    return WeightPatch(
        patch_id="p-standard",
        base_model_id="toy",
        base_model_revision="r1",
        base_model_fingerprint=fp,
        deltas={"fc1.weight": delta},
        hypothesis="standard valid patch",
    )


@pytest.fixture
def strict_budget() -> PatchBudget:
    return PatchBudget(
        max_changed_tensors=2,
        max_l2_norm=5.0,
        max_relative_norm=1.0,
        allowed_tensor_names=["fc1.weight", "fc2.weight"],
    )


def test_budget_omission_fails_closed_in_sandboxed(clean_model: ToyMLP, standard_patch: WeightPatch):
    """Calling apply_patch_sandboxed with budget=None and require_budget=True must raise BudgetViolationError."""
    with pytest.raises(BudgetViolationError, match="A non-null PatchBudget is required"):
        with apply_patch_sandboxed(clean_model, standard_patch, budget=None, require_budget=True):
            pass


def test_budget_omission_fails_closed_in_copy(clean_model: ToyMLP, standard_patch: WeightPatch):
    """Calling apply_patch_to_copy with budget=None and require_budget=True must raise BudgetViolationError."""
    with pytest.raises(BudgetViolationError, match="A non-null PatchBudget is required"):
        apply_patch_to_copy(clean_model, standard_patch, budget=None, require_budget=True)


def test_missing_fingerprint_rejected_when_required(clean_model: ToyMLP, strict_budget: PatchBudget):
    """When require_fingerprint=True, a patch without base_model_fingerprint must be rejected."""
    delta = np.ones((6, 4), dtype=np.float32) * 0.05
    patch_no_fp = WeightPatch(
        patch_id="no-fp",
        base_model_id="toy",
        base_model_revision="r1",
        base_model_fingerprint=None,
        deltas={"fc1.weight": delta},
        hypothesis="missing fingerprint",
    )
    with pytest.raises(BaseModelMismatchError, match="lacks required base_model_fingerprint"):
        validate_patch_against_budget(clean_model, patch_no_fp, strict_budget, require_fingerprint=True)

    with pytest.raises(BaseModelMismatchError, match="lacks required base_model_fingerprint"):
        with apply_patch_sandboxed(clean_model, patch_no_fp, budget=strict_budget, require_fingerprint=True):
            pass


def test_mismatched_fingerprint_rejected(clean_model: ToyMLP, strict_budget: PatchBudget):
    """A patch declared for a different model checkpoint must be rejected."""
    delta = np.ones((6, 4), dtype=np.float32) * 0.05
    patch_wrong_fp = WeightPatch(
        patch_id="wrong-fp",
        base_model_id="toy",
        base_model_revision="r1",
        base_model_fingerprint="f" * 64,
        deltas={"fc1.weight": delta},
        hypothesis="mismatched fingerprint",
    )
    with pytest.raises(BaseModelMismatchError, match="fingerprint mismatch"):
        validate_patch_against_budget(clean_model, patch_wrong_fp, strict_budget)


def test_incompatible_dtype_rejected(clean_model: ToyMLP, strict_budget: PatchBudget):
    """Patches with non-floating dtypes (e.g. int32 or object) must fail closed with DtypeMismatchError."""
    delta_int = np.ones((6, 4), dtype=np.int32)
    patch_int = WeightPatch(
        patch_id="int-patch",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={"fc1.weight": delta_int},
        hypothesis="integer delta test",
    )
    with pytest.raises(DtypeMismatchError, match="incompatible non-floating dtype"):
        validate_patch_against_budget(clean_model, patch_int, strict_budget)


def test_nan_and_inf_rejected_before_mutation(clean_model: ToyMLP, strict_budget: PatchBudget):
    """Patches containing NaN or Inf must be rejected without mutating any weight."""
    orig_w = clean_model.fc1.weight.clone()

    nan_delta = np.ones((6, 4), dtype=np.float32)
    nan_delta[0, 0] = np.nan
    patch_nan = WeightPatch(
        patch_id="nan-patch",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={"fc1.weight": nan_delta},
        hypothesis="nan rejection",
    )
    with pytest.raises(BudgetViolationError, match="non-finite"):
        with apply_patch_sandboxed(clean_model, patch_nan, budget=strict_budget):
            pass

    assert torch.equal(clean_model.fc1.weight, orig_w)


def test_shape_mismatch_rejected(clean_model: ToyMLP, strict_budget: PatchBudget):
    """Patches with incompatible tensor shapes must raise ValueError before mutation."""
    delta_wrong_shape = np.ones((6, 5), dtype=np.float32)
    patch_wrong_shape = WeightPatch(
        patch_id="shape-patch",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={"fc1.weight": delta_wrong_shape},
        hypothesis="wrong shape test",
    )
    with pytest.raises(ValueError, match="does not match parameter shape"):
        validate_patch_against_budget(clean_model, patch_wrong_shape, strict_budget)


def test_unapproved_tensor_rejected_by_allowlist(clean_model: ToyMLP):
    """Patches modifying tensors not on the allowlist must raise AllowlistViolationError."""
    budget = PatchBudget(allowed_tensor_names=["fc2.weight"])
    delta = np.ones((6, 4), dtype=np.float32) * 0.01
    patch = WeightPatch(
        patch_id="unapproved-patch",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={"fc1.weight": delta},
        hypothesis="unapproved layer",
    )
    with pytest.raises(AllowlistViolationError, match="not on allowlist"):
        validate_patch_against_budget(clean_model, patch, budget)


def test_cryptographic_binding_hash_distinctiveness(clean_model: ToyMLP):
    """Binding hash must bind checkpoint fingerprint, tensor names, shapes, and dtypes."""
    fp1 = compute_model_fingerprint(clean_model)
    delta1 = np.ones((6, 4), dtype=np.float32) * 0.05
    delta2 = np.ones((6, 4), dtype=np.float64) * 0.05  # different dtype

    patch1 = WeightPatch(
        patch_id="p1",
        base_model_id="toy",
        base_model_revision="r1",
        base_model_fingerprint=fp1,
        deltas={"fc1.weight": delta1},
        hypothesis="test binding",
    )
    patch2 = WeightPatch(
        patch_id="p2",
        base_model_id="toy",
        base_model_revision="r1",
        base_model_fingerprint=fp1,
        deltas={"fc1.weight": delta2},
        hypothesis="test binding",
    )

    h1 = patch1.compute_binding_hash()
    h2 = patch2.compute_binding_hash()

    assert len(h1) == 64
    assert len(h2) == 64
    assert h1 != h2  # different delta dtype and patch_id produce different binding hash


def test_simulated_rollback_checksum_mismatch(clean_model: ToyMLP, strict_budget: PatchBudget):
    """If an unauthorized in-place mutation alters backup during execution, rollback must raise RuntimeError."""
    delta = np.ones((6, 4), dtype=np.float32) * 0.01
    patch = WeightPatch(
        patch_id="p-sim-crash",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={"fc1.weight": delta},
        hypothesis="rollback checksum test",
    )

    # Monkeypatch _hash_tensor during exit to simulate a rollback checksum mismatch
    import weightliftai.weight_programming.sandbox as sandbox_mod
    original_hash_fn = sandbox_mod._hash_tensor

    try:
        with pytest.raises(RuntimeError, match="Rollback failed for 'fc1.weight': hash mismatch"):
            with apply_patch_sandboxed(clean_model, patch, budget=strict_budget):
                # Replace hash function inside block to simulate mismatch on restoration
                sandbox_mod._hash_tensor = lambda param: "tampered_checksum_0000000000000000"
    finally:
        sandbox_mod._hash_tensor = original_hash_fn
