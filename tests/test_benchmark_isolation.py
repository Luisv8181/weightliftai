import gc
import numpy as np
import pytest
import torch

from weightliftai.weight_programming.proposer_api import (
    BlindProposerEnvironment,
    EvaluationBudgetExceededError,
    ProposerIsolationViolationError,
)
from weightliftai.weight_programming.schemas import WeightPatch


def test_proposer_has_no_secrets_in_attributes():
    env = BlindProposerEnvironment(seed=42)

    # 1. Negative assertion on attribute existence
    assert not hasattr(env, "_sealed_clean_fc2")
    assert not hasattr(env, "_sealed_held_out")
    assert not hasattr(env, "_sealed_ood")
    assert not hasattr(env, "_sealed_regression")
    assert not hasattr(env, "evaluator")

    # 2. Accessing forbidden secrets raises ProposerIsolationViolationError
    with pytest.raises(ProposerIsolationViolationError, match="Access denied"):
        _ = env._sealed_clean_fc2

    with pytest.raises(ProposerIsolationViolationError, match="Access denied"):
        _ = env._sealed_held_out

    with pytest.raises(ProposerIsolationViolationError, match="Access denied"):
        _ = env.clean_model


def test_object_graph_contains_no_sealed_splits():
    env = BlindProposerEnvironment(seed=42)
    # Collect all dictionaries currently in memory
    gc.collect()
    all_dicts = [o for o in gc.get_objects() if isinstance(o, dict)]

    # Verify that none of them contain secret keys used in evaluator splits
    for d in all_dicts:
        assert "_sealed_held_out" not in d
        assert "_sealed_clean_fc2" not in d
        assert "_sealed_ood" not in d
        assert "_sealed_regression" not in d


def test_subprocess_evaluation_execution():
    env = BlindProposerEnvironment(seed=42)
    report = env.get_diagnostic_failure_report()
    assert report["observed_diagnostic_mse"] > 1.0

    # Build a valid candidate patch
    corrupted = env.get_corrupted_model()
    # Compute empirical gradient on diagnostic data only
    opt = torch.optim.Adam([corrupted.fc2.weight], lr=0.05)
    init_w = corrupted.fc2.weight.data.clone()
    for _ in range(50):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(
            corrupted(report["inputs"])[:, :1], report["expected_outputs"]
        )
        loss.backward()
        if corrupted.fc2.weight.grad is not None:
            corrupted.fc2.weight.grad[1, :] = 0.0
        opt.step()

    delta_arr = (corrupted.fc2.weight.data - init_w).cpu().numpy()

    candidate = WeightPatch(
        patch_id="candidate-empirical-test",
        base_model_id="toy-mlp-ground-truth",
        base_model_revision="corrupted-v1",
        base_model_fingerprint=env.corrupted_fingerprint,
        deltas={"fc2.weight": delta_arr},
        hypothesis="Empirical gradient repair in isolated subprocess.",
        proposed_by="test_proposer",
    )

    result = env.submit_candidate_patch(candidate)
    assert result["success"] is True
    assert result["passed_gates"] is True
    assert result["gains"]["target_held_out_gain"] > 0.5
    assert "aggregate_scores" in result

    # Verify per-sample leakage prevention: no individual inputs/labels/predictions returned
    assert "inputs" not in result
    assert "predictions" not in result
    assert "losses_per_sample" not in result


def test_query_budget_enforcement():
    env = BlindProposerEnvironment(seed=42, max_queries=1)
    delta_arr = np.zeros((2, 8), dtype=np.float32)

    candidate = WeightPatch(
        patch_id="null-patch-budget-test",
        base_model_id="toy-mlp-ground-truth",
        base_model_revision="corrupted-v1",
        base_model_fingerprint=env.corrupted_fingerprint,
        deltas={"fc2.weight": delta_arr},
        hypothesis="Null patch for budget testing.",
        proposed_by="test_proposer",
    )

    # First query succeeds
    result1 = env.submit_candidate_patch(candidate)
    assert result1["success"] is True

    # Second query is strictly blocked by query budget
    with pytest.raises(EvaluationBudgetExceededError, match="Evaluation query budget exceeded"):
        env.submit_candidate_patch(candidate)


def test_fingerprint_mismatch_rejected_before_subprocess():
    env = BlindProposerEnvironment(seed=42)
    delta_arr = np.zeros((2, 8), dtype=np.float32)

    candidate = WeightPatch(
        patch_id="mismatch-patch-test",
        base_model_id="toy-mlp-ground-truth",
        base_model_revision="corrupted-v1",
        base_model_fingerprint="wrong-fingerprint-" + "0" * 48,
        deltas={"fc2.weight": delta_arr},
        hypothesis="Mismatch test.",
        proposed_by="test_proposer",
    )

    with pytest.raises(ValueError, match="Candidate patch fingerprint mismatch"):
        env.submit_candidate_patch(candidate)
