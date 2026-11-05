"""Tests for Weight-Space Programming, parameter budgets, sandboxing, and defect recovery."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from weightliftai.weight_programming import (
    AllowlistViolationError,
    BaseModelMismatchError,
    BudgetViolationError,
    IndependentEvaluator,
    PatchBudget,
    ToyMLP,
    WeightPatch,
    apply_patch_sandboxed,
    apply_patch_to_copy,
    compute_model_fingerprint,
    hash_tensor_bytes,
    inspect_model_parameters,
    run_known_defect_experiment,
    validate_patch_against_budget,
)


def test_patch_schema_and_sha256():
    patch = WeightPatch(
        patch_id="p-001",
        base_model_id="test-model",
        base_model_revision="rev-1",
        deltas={"layer.weight": np.zeros((4, 4), dtype=np.float32)},
        hypothesis="Test patch hypothesis",
    )
    s = patch.sha256()
    assert len(s) == 64
    summary = patch.summary()
    assert summary["patch_id"] == "p-001"
    assert summary["num_tensors_modified"] == 1


def test_inspect_model_parameters():
    model = ToyMLP(in_features=4, hidden_dim=6, out_features=2)
    report = inspect_model_parameters(model)
    assert "fc1.weight" in report
    assert report["fc1.weight"]["shape"] == [6, 4]
    assert report["fc1.weight"]["numel"] == 24
    assert len(report["fc1.weight"]["sha256"]) == 64


def test_budget_validation_and_violations():
    model = ToyMLP(in_features=4, hidden_dim=6, out_features=2)
    delta_fc1 = np.ones((6, 4), dtype=np.float32) * 0.01
    
    # 1. Valid patch
    patch = WeightPatch(
        patch_id="valid",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={"fc1.weight": delta_fc1},
        hypothesis="safe",
    )
    budget = PatchBudget(
        max_changed_tensors=2,
        max_l2_norm=1.0,
        max_relative_norm=0.5,
        allowed_tensor_names=["fc1.weight", "fc2.weight"],
    )
    res = validate_patch_against_budget(model, patch, budget)
    assert res["status"] == "passed"

    # 2. Allowlist violation
    budget_strict = PatchBudget(allowed_tensor_names=["fc2.weight"])
    with pytest.raises(AllowlistViolationError):
        validate_patch_against_budget(model, patch, budget_strict)

    # 3. L2 norm budget violation
    big_delta = np.ones((6, 4), dtype=np.float32) * 100.0
    patch_huge = WeightPatch(
        patch_id="huge",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={"fc1.weight": big_delta},
        hypothesis="huge",
    )
    with pytest.raises(BudgetViolationError):
        validate_patch_against_budget(model, patch_huge, budget)


def test_budget_rejects_nan_and_inf():
    model = ToyMLP(in_features=4, hidden_dim=6, out_features=2)
    budget = PatchBudget(max_l2_norm=10.0, max_relative_norm=1.0)

    # NaN in delta
    nan_delta = np.ones((6, 4), dtype=np.float32)
    nan_delta[0, 0] = np.nan
    patch_nan = WeightPatch(
        patch_id="nan-patch",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={"fc1.weight": nan_delta},
        hypothesis="nan test",
    )
    with pytest.raises(BudgetViolationError, match="non-finite"):
        validate_patch_against_budget(model, patch_nan, budget)

    # Inf in delta
    inf_delta = np.ones((6, 4), dtype=np.float32)
    inf_delta[0, 0] = np.inf
    patch_inf = WeightPatch(
        patch_id="inf-patch",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={"fc1.weight": inf_delta},
        hypothesis="inf test",
    )
    with pytest.raises(BudgetViolationError, match="non-finite"):
        validate_patch_against_budget(model, patch_inf, budget)


def test_base_model_fingerprint_verification():
    model = ToyMLP(in_features=4, hidden_dim=6, out_features=2)
    correct_fp = compute_model_fingerprint(model)
    budget = PatchBudget(max_l2_norm=5.0, max_relative_norm=1.0)
    delta = np.ones((6, 4), dtype=np.float32) * 0.01

    # Matching fingerprint passes
    patch_ok = WeightPatch(
        patch_id="fp-ok",
        base_model_id="toy",
        base_model_revision="r1",
        base_model_fingerprint=correct_fp,
        deltas={"fc1.weight": delta},
        hypothesis="fingerprint match test",
    )
    res = validate_patch_against_budget(model, patch_ok, budget)
    assert res["status"] == "passed"

    # Mismatched fingerprint raises BaseModelMismatchError
    patch_bad = WeightPatch(
        patch_id="fp-bad",
        base_model_id="toy",
        base_model_revision="r1",
        base_model_fingerprint="0000000000000000000000000000000000000000000000000000000000000000",
        deltas={"fc1.weight": delta},
        hypothesis="fingerprint mismatch test",
    )
    with pytest.raises(BaseModelMismatchError):
        validate_patch_against_budget(model, patch_bad, budget)


def test_patch_rejects_empty_deltas():
    model = ToyMLP(in_features=4, hidden_dim=6, out_features=2)
    budget = PatchBudget()
    patch_empty = WeightPatch(
        patch_id="empty",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={},
        hypothesis="empty test",
    )
    with pytest.raises(BudgetViolationError, match="no parameter deltas"):
        validate_patch_against_budget(model, patch_empty, budget)


def test_tied_weight_aliasing_conflict():
    model = ToyMLP(in_features=4, hidden_dim=6, out_features=2)
    # Tie fc2.weight to fc1.weight (simulate tied embeddings / head)
    # Give them compatible shapes first
    model.fc2 = torch.nn.Linear(6, 4, bias=False)
    model.fc2.weight = model.fc1.weight  # tied storage!
    
    delta = np.ones((6, 4), dtype=np.float32) * 0.01
    budget = PatchBudget(max_l2_norm=5.0, max_relative_norm=1.0)

    # Modifying only one tied weight raises AllowlistViolationError
    patch_tied = WeightPatch(
        patch_id="tied-patch",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={"fc1.weight": delta},
        hypothesis="tied test",
    )
    with pytest.raises(AllowlistViolationError, match="Tied-parameter conflict"):
        validate_patch_against_budget(model, patch_tied, budget)


def test_apply_patch_sandboxed_reversible_rollback():
    model = ToyMLP(in_features=4, hidden_dim=6, out_features=2)
    orig_w = model.fc1.weight.clone()
    orig_w_arr = orig_w.detach().cpu().numpy()
    
    delta = np.ones((6, 4), dtype=np.float32) * 0.5
    patch = WeightPatch(
        patch_id="p-sandbox",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={"fc1.weight": delta},
        hypothesis="sandbox test",
    )

    with apply_patch_sandboxed(model, patch) as patched_model:
        # Inside sandbox: weights must reflect delta
        curr_w = patched_model.fc1.weight.detach().cpu().numpy()
        np.testing.assert_allclose(curr_w, orig_w_arr + delta, rtol=1e-5)

    # Outside sandbox: weights must be bit-exact restored
    restored_w = model.fc1.weight.detach().cpu().numpy()
    np.testing.assert_array_equal(restored_w, orig_w_arr)


def test_rollback_preserves_state_on_unhandled_exception():
    model = ToyMLP(in_features=4, hidden_dim=6, out_features=2)
    orig_w_arr = model.fc1.weight.detach().cpu().numpy().copy()
    
    delta = np.ones((6, 4), dtype=np.float32) * 0.5
    patch = WeightPatch(
        patch_id="p-crash",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={"fc1.weight": delta},
        hypothesis="crash test",
    )

    with pytest.raises(RuntimeError, match="Simulated execution crash"):
        with apply_patch_sandboxed(model, patch) as patched_model:
            # Crash occurs inside context block
            raise RuntimeError("Simulated execution crash")

    # Weights must be bit-exact restored even after exception
    np.testing.assert_array_equal(model.fc1.weight.detach().cpu().numpy(), orig_w_arr)


def test_native_byte_hashing_distinguishes_dtypes():
    p1 = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.float32))
    p2 = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.float16))
    
    h1 = hash_tensor_bytes(p1)
    h2 = hash_tensor_bytes(p2)
    
    # Must produce different hashes due to dtype and storage byte representation
    assert h1 != h2
    assert len(h1) == 64
    assert len(h2) == 64


def test_apply_patch_to_copy_leaves_original_untouched():
    model = ToyMLP(in_features=4, hidden_dim=6, out_features=2)
    orig_w_arr = model.fc1.weight.detach().cpu().numpy().copy()
    
    delta = np.ones((6, 4), dtype=np.float32) * 0.25
    patch = WeightPatch(
        patch_id="p-clone",
        base_model_id="toy",
        base_model_revision="r1",
        deltas={"fc1.weight": delta},
        hypothesis="clone test",
    )

    cloned = apply_patch_to_copy(model, patch)
    
    # Original is untouched
    np.testing.assert_array_equal(model.fc1.weight.detach().cpu().numpy(), orig_w_arr)
    # Clone is modified
    np.testing.assert_allclose(cloned.fc1.weight.detach().cpu().numpy(), orig_w_arr + delta, rtol=1e-5)


def test_run_known_defect_experiment():
    result = run_known_defect_experiment()
    assert result["passed_gates"] is True
    assert result["recommendation"] == "promote"
    assert result["gains"]["target_held_out_gain"] > 0.0
