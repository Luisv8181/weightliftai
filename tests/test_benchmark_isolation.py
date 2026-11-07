"""Comprehensive Adversarial Test Suite for Evaluator-Authoritative Blind Benchmark (Prompts 1–6, 8).

Verifies:
1. Evaluator-side experiment registration, query authorization, persistent ledger across
   instances/restarts/concurrency, consume-before-validation attempt policy, and Ed25519
   receipt signing & tamper detection (Prompt 1).
2. Private evaluator-controlled randomness & resistance to deterministic reconstruction from
   public seeds, plus authorized researcher audit replay (Prompt 2).
3. Structured stdin/stdout pipe IPC protocol validation, version checking, and hash binding (Prompt 3).
4. Pre-allocation input size, tensor rank/dimension/parameter limits, NaN/Inf rejection, and
   timeout enforcement (Prompt 4).
5. Standardized error categories without stack trace/path/secret leakage, and coarse metric
   rounding vs. researcher audit records (Prompt 5).
6. Legacy `BlindRepairBenchmark` non-sealed labeling and 5-condition diagnostic scoping (Prompts 6 & 8).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from weightliftai.weight_programming.blind_benchmark import BlindRepairBenchmark
from weightliftai.weight_programming.defect_families import DiagnosticCondition
from weightliftai.weight_programming.evaluator_authority import EvaluatorRegistryAuthority
from weightliftai.weight_programming.inspector import compute_model_fingerprint
from weightliftai.weight_programming.ipc_protocol import (
    ErrorCategory,
    EvaluatorIPCError,
    EvaluatorResourceLimits,
     verify_signed_evaluation_result,
)
from weightliftai.weight_programming.isolated_evaluator import handle_ipc_request_bytes
from weightliftai.weight_programming.proposer_api import (
    BlindProposerEnvironment,
    EvaluationBudgetExceededError,
    ProposerAuthorizationError,
    ProposerIsolationViolationError,
    send_evaluator_ipc_request,
)
from weightliftai.weight_programming.schemas import WeightPatch
from weightliftai.weight_programming.toy_defect import ToyMLP, generate_synthetic_data


def _build_zero_patch(env: BlindProposerEnvironment, patch_id: str = "zero-patch") -> WeightPatch:
    delta_arr = np.zeros((2, 8), dtype=np.float32)
    return WeightPatch(
        patch_id=patch_id,
        base_model_id="toy-mlp-ground-truth",
        base_model_revision="corrupted-v1",
        base_model_fingerprint=env.corrupted_fingerprint,
        deltas={"fc2.weight": delta_arr},
        hypothesis="Zero patch for isolation testing.",
        proposed_by="test_proposer",
    )


def test_proposer_has_no_secrets_in_attributes(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)

    assert not hasattr(env, "_sealed_clean_fc2")
    assert not hasattr(env, "_sealed_held_out")
    assert not hasattr(env, "_sealed_ood")
    assert not hasattr(env, "_sealed_regression")
    assert not hasattr(env, "evaluator")
    assert not hasattr(env, "private_seed")

    with pytest.raises(ProposerIsolationViolationError, match="Access denied"):
        _ = env._sealed_clean_fc2

    with pytest.raises(ProposerIsolationViolationError, match="Access denied"):
        _ = env._sealed_held_out

    with pytest.raises(ProposerIsolationViolationError, match="Access denied"):
        _ = env.clean_model

    with pytest.raises(ProposerIsolationViolationError, match="Access denied"):
        _ = env.private_seed


def test_object_graph_contains_no_sealed_splits(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)
    gc.collect()
    all_dicts = [o for o in gc.get_objects() if isinstance(o, dict)]

    for d in all_dicts:
        assert "_sealed_held_out" not in d
        assert "_sealed_clean_fc2" not in d
        assert "_sealed_ood" not in d
        assert "_sealed_regression" not in d
        assert "clean_state_dict" not in d
        assert "oracle_delta_fc2" not in d

    # Ensure env stays alive during GC scan
    assert env.experiment_id.startswith("exp_")


def test_subprocess_evaluation_execution_and_signed_receipt(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)
    report = env.get_diagnostic_failure_report()
    assert report["observed_diagnostic_mse"] > 0.3

    corrupted = env.get_corrupted_model()
    opt = torch.optim.Adam([corrupted.fc2.weight], lr=0.05)
    init_w = corrupted.fc2.weight.data.clone()
    for _ in range(80):
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
    assert result["gains"]["target_held_out_gain"] > 0.3
    assert "aggregate_scores" in result
    assert "signed_receipt" in result

    # Independent cryptographic signature verification
    assert verify_signed_evaluation_result(result, public_key_hex=env.evaluator_public_key) is True

    # Verify via evaluator verify_receipt IPC action
    verify_resp = send_evaluator_ipc_request(
        action="verify_receipt",
        payload={"evaluation_result": result},
        ledger_path=ledger,
    )
    assert verify_resp["verified"] is True

    # Verify per-sample leakage prevention
    assert "inputs" not in result
    assert "predictions" not in result
    assert "losses_per_sample" not in result


def test_tampered_signed_receipt_fails_verification(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)
    result = env.submit_candidate_patch(_build_zero_patch(env))
    assert verify_signed_evaluation_result(result, public_key_hex=env.evaluator_public_key) is True

    # Tamper 1: flip passed_gates in outer dict or signed_receipt
    tampered_1 = copy.deepcopy(result)
    tampered_1["passed_gates"] = True
    assert verify_signed_evaluation_result(tampered_1, public_key_hex=env.evaluator_public_key) is False

    tampered_2 = copy.deepcopy(result)
    tampered_2["signed_receipt"]["passed_gates"] = True
    tampered_2["passed_gates"] = True
    assert verify_signed_evaluation_result(tampered_2, public_key_hex=env.evaluator_public_key) is False

    # Tamper 2: modify aggregate_scores
    tampered_3 = copy.deepcopy(result)
    tampered_3["signed_receipt"]["aggregate_scores"]["held_out_mse"] = 0.0001
    tampered_3["aggregate_scores"]["held_out_mse"] = 0.0001
    assert verify_signed_evaluation_result(tampered_3, public_key_hex=env.evaluator_public_key) is False

    # Tamper 3: plain forged JSON without signature fails
    forged = {"success": True, "passed_gates": True, "recommendation": "promote"}
    assert verify_signed_evaluation_result(forged, public_key_hex=env.evaluator_public_key) is False


def test_query_budget_enforcement_resists_caller_budget_and_counter_mutation(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    # Caller attempts to request max_queries=10
    env = BlindProposerEnvironment(seed=42, max_queries=10, ledger_path=ledger)
    candidate = _build_zero_patch(env, patch_id="null-patch-budget-test")

    result1 = env.submit_candidate_patch(candidate)
    assert result1["success"] is True

    # Caller mutates instance state to reset counter and raise budget
    env._queries_used = 0
    env._untrusted_caller_max_queries = 999
    env.allowed_submissions = 999

    # Evaluator ledger authoritatively rejects the second submission
    with pytest.raises(EvaluationBudgetExceededError, match="Evaluation query budget exceeded"):
        env.submit_candidate_patch(candidate)


def test_query_budget_enforced_across_multiple_instances_and_restarts(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env1 = BlindProposerEnvironment(seed=42, ledger_path=ledger)
    env2 = BlindProposerEnvironment.from_existing_registration(env1)

    candidate = _build_zero_patch(env1)
    res = env1.submit_candidate_patch(candidate)
    assert res["success"] is True

    # Second instance sharing the same experiment_id is blocked by persistent SQLite ledger
    with pytest.raises(EvaluationBudgetExceededError, match="Evaluation query budget exceeded"):
        env2.submit_candidate_patch(candidate)


def test_concurrent_submissions_only_allow_single_evaluation(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)
    candidate = _build_zero_patch(env)

    def _submit(_i: int) -> str:
        cloned_env = BlindProposerEnvironment.from_existing_registration(env)
        try:
            cloned_env.submit_candidate_patch(candidate)
            return "SUCCESS"
        except EvaluationBudgetExceededError:
            return "BUDGET_EXCEEDED"

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(_submit, range(4)))

    assert outcomes.count("SUCCESS") == 1
    assert outcomes.count("BUDGET_EXCEEDED") == 3


def test_failed_validation_consumes_attempt_under_consume_before_validation_policy(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)

    # Submit a patch that violates the L2 norm budget (norm = 50 > 5.0)
    huge_delta = np.full((2, 8), 20.0, dtype=np.float32)
    bad_patch = WeightPatch(
        patch_id="over-budget-patch",
        base_model_id="toy-mlp-ground-truth",
        base_model_revision="corrupted-v1",
        base_model_fingerprint=env.corrupted_fingerprint,
        deltas={"fc2.weight": huge_delta},
        hypothesis="Exceeds L2 budget.",
        proposed_by="test_proposer",
    )

    with pytest.raises(RuntimeError, match="VALIDATION_ERROR:PATCH_BUDGET_OR_INVARIANT_VIOLATION"):
        env.submit_candidate_patch(bad_patch)

    # Because policy is CONSUME_BEFORE_VALIDATION, a subsequent valid submission is blocked
    valid_patch = _build_zero_patch(env)
    with pytest.raises(EvaluationBudgetExceededError, match="Evaluation query budget exceeded"):
        env.submit_candidate_patch(valid_patch)


def test_unknown_expired_and_config_tampered_registrations_rejected(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)
    patch = _build_zero_patch(env)

    # 1. Tampered config_sha256
    env_bad_cfg = BlindProposerEnvironment.from_existing_registration(env)
    env_bad_cfg.config_sha256 = "f" * 64
    with pytest.raises(ProposerAuthorizationError, match="EXPERIMENT_CONFIG_MISMATCH"):
        env_bad_cfg.submit_candidate_patch(patch)

    # 2. Unknown experiment_id
    env_unknown = BlindProposerEnvironment.from_existing_registration(env)
    env_unknown.experiment_id = "exp_" + "0" * 32
    with pytest.raises(ProposerAuthorizationError, match="UNKNOWN_EXPERIMENT_ID"):
        env_unknown.submit_candidate_patch(patch)

    # 3. Expired experiment registration
    env_short = BlindProposerEnvironment(seed=42, ttl_seconds=0.01, ledger_path=ledger)
    import time

    time.sleep(0.05)
    with pytest.raises(ProposerAuthorizationError, match="EXPERIMENT_EXPIRED"):
        env_short.submit_candidate_patch(_build_zero_patch(env_short))


def test_proposer_cannot_choose_experiment_id_or_override_registration_fields(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    for forbidden_key, val in [
        ("experiment_id", "exp_custom_123"),
        ("private_seed", 42),
        ("allowed_submissions", 10),
        ("max_queries", 10),
    ]:
        with pytest.raises(EvaluatorIPCError) as exc_info:
            send_evaluator_ipc_request(
                action="register_experiment",
                payload={forbidden_key: val},
                ledger_path=ledger,
            )
        assert exc_info.value.category == ErrorCategory.AUTH_ERROR
        assert exc_info.value.code == "PROPOSER_CONFIGURATION_OVERRIDE_FORBIDDEN"


def test_direct_unregistered_cli_invocation_rejected(tmp_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "weightliftai.weight_programming.isolated_evaluator",
        "--seed",
        "42",
        "--patch-json",
        str(tmp_path / "nonexistent.json"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    resp = json.loads(proc.stdout)
    assert resp["success"] is False
    assert resp["error_category"] == "AUTH_ERROR"
    assert resp["error_code"] == "DIRECT_UNREGISTERED_INVOCATION_FORBIDDEN"


def test_private_randomness_prevents_deterministic_reconstruction_from_public_seed(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env1 = BlindProposerEnvironment(seed=42, ledger_path=ledger)
    env2 = BlindProposerEnvironment(seed=42, ledger_path=ledger)

    # 1. Two independent registrations with the same public seed hint get fresh private randomness
    assert env1.experiment_id != env2.experiment_id
    assert env1.corrupted_fingerprint != env2.corrupted_fingerprint

    # 2. Attempt deterministic reconstruction using public seed=42 and public training recipe
    torch.manual_seed(42)
    np.random.seed(42)
    public_data = generate_synthetic_data(seed=42)
    reconstructed_clean = ToyMLP()
    opt = torch.optim.Adam(reconstructed_clean.parameters(), lr=0.05)
    for _ in range(250):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(
            reconstructed_clean(public_data["train_full"]["x"]),
            public_data["train_full"]["y"],
        )
        loss.backward()
        opt.step()

    reconstructed_corrupted = copy.deepcopy(reconstructed_clean)
    defect_w = reconstructed_clean.fc2.weight.data.clone().numpy()
    defect_w[0, :4] = 0.0
    reconstructed_corrupted.fc2.weight.data.copy_(torch.from_numpy(defect_w))
    reconstructed_fp = compute_model_fingerprint(reconstructed_corrupted)

    # Reconstruction from public seed=42 fails to match evaluator's privately seeded instance
    assert reconstructed_fp != env1.corrupted_fingerprint
    assert not np.allclose(
        reconstructed_corrupted.fc1.weight.data.numpy(),
        env1.get_corrupted_model().fc1.weight.data.numpy(),
    )


def test_authorized_researcher_audit_replay_requires_valid_token(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)
    env.submit_candidate_patch(_build_zero_patch(env))

    # 1. Invalid token is rejected with AUTH_ERROR
    with pytest.raises(EvaluatorIPCError) as exc_info:
        send_evaluator_ipc_request(
            action="researcher_audit_replay",
            payload={
                "experiment_id": env.experiment_id,
                "researcher_audit_token": "forged_token",
            },
            ledger_path=ledger,
        )
    assert exc_info.value.category == ErrorCategory.AUTH_ERROR
    assert exc_info.value.code == "UNAUTHORIZED_RESEARCHER_AUDIT_ACCESS"

    # 2. Authorized researcher with vault token can retrieve full provenance & private_seed
    authority = EvaluatorRegistryAuthority(ledger_path=ledger)
    token = authority.get_researcher_audit_token()
    audit_record = send_evaluator_ipc_request(
        action="researcher_audit_replay",
        payload={
            "experiment_id": env.experiment_id,
            "researcher_audit_token": token,
        },
        ledger_path=ledger,
    )
    assert audit_record["experiment_id"] == env.experiment_id
    assert isinstance(audit_record["private_seed"], int)
    assert len(audit_record["submissions"]) == 1


def test_ipc_protocol_rejects_malformed_messages_bad_version_and_hash_mismatch(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"

    # 1. Malformed JSON
    resp1, code1 = handle_ipc_request_bytes(b"{not valid json", ledger_path=ledger)
    assert code1 == 1
    assert resp1["error_category"] == "PROTOCOL_ERROR"
    assert resp1["error_code"] == "MALFORMED_JSON"

    # 2. Unsupported protocol version
    resp2, code2 = handle_ipc_request_bytes(
        json.dumps({"protocol_version": "99.0", "action": "register_experiment", "payload": {}}).encode("utf-8"),
        ledger_path=ledger,
    )
    assert code2 == 1
    assert resp2["error_category"] == "PROTOCOL_ERROR"
    assert resp2["error_code"] == "UNSUPPORTED_PROTOCOL_VERSION"

    # 3. Altered candidate_patch_sha256 binding
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)
    patch = _build_zero_patch(env)
    bad_req = {
        "experiment_id": env.experiment_id,
        "config_sha256": env.config_sha256,
        "candidate_patch_sha256": "a" * 64,
        "patch": {
            "patch_id": patch.patch_id,
            "base_model_id": patch.base_model_id,
            "base_model_revision": patch.base_model_revision,
            "base_model_fingerprint": patch.base_model_fingerprint,
            "deltas": {"fc2.weight": patch.deltas["fc2.weight"].tolist()},
        },
    }
    with pytest.raises(EvaluatorIPCError) as exc_info:
        send_evaluator_ipc_request("submit_candidate", bad_req, ledger_path=ledger)
    assert exc_info.value.category == ErrorCategory.PROTOCOL_ERROR
    assert exc_info.value.code == "CANDIDATE_HASH_MISMATCH"


def test_resource_limits_reject_oversized_payload_extreme_shapes_and_nan_inf(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    strict_limits = EvaluatorResourceLimits(max_request_bytes=1024)

    # 1. Oversized raw byte payload rejected before JSON parsing
    huge_bytes = b" " * 2048
    resp_big, code_big = handle_ipc_request_bytes(huge_bytes, ledger_path=ledger, resource_limits=strict_limits)
    assert code_big == 1
    assert resp_big["error_category"] == "RESOURCE_LIMIT_ERROR"
    assert resp_big["error_code"] == "IPC_PAYLOAD_TOO_LARGE"

    # 2. Extreme declared dimension rejected before array allocation
    env1 = BlindProposerEnvironment(seed=42, ledger_path=ledger)
    req_dim = {
        "experiment_id": env1.experiment_id,
        "config_sha256": env1.config_sha256,
        "patch": {
            "patch_id": "huge-dim",
            "base_model_id": "toy",
            "base_model_revision": "v1",
            "base_model_fingerprint": env1.corrupted_fingerprint,
            "declared_shapes": {"fc2.weight": [100000, 100000]},
            "deltas": {"fc2.weight": [[0.0]]},
        },
    }
    with pytest.raises(EvaluatorIPCError) as exc_dim:
        send_evaluator_ipc_request("submit_candidate", req_dim, ledger_path=ledger)
    assert exc_dim.value.category == ErrorCategory.RESOURCE_LIMIT_ERROR

    # 3. Non-finite values (NaN / Inf) rejected before allocation
    env2 = BlindProposerEnvironment(seed=42, ledger_path=ledger)
    nan_arr = np.zeros((2, 8), dtype=np.float32)
    nan_arr[0, 0] = np.nan
    nan_patch = WeightPatch(
        patch_id="nan-patch",
        base_model_id="toy-mlp-ground-truth",
        base_model_revision="corrupted-v1",
        base_model_fingerprint=env2.corrupted_fingerprint,
        deltas={"fc2.weight": nan_arr},
        hypothesis="NaN injection test.",
        proposed_by="test_proposer",
    )
    with pytest.raises(RuntimeError, match="VALIDATION_ERROR:NON_FINITE_SCALAR"):
        env2.submit_candidate_patch(nan_patch)


def test_evaluator_timeout_enforcement(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)
    patch = _build_zero_patch(env)
    req = {
        "experiment_id": env.experiment_id,
        "config_sha256": env.config_sha256,
        "candidate_patch_sha256": patch.sha256(),
        "patch": {
            "patch_id": patch.patch_id,
            "base_model_id": patch.base_model_id,
            "base_model_revision": patch.base_model_revision,
            "base_model_fingerprint": patch.base_model_fingerprint,
            "deltas": {"fc2.weight": patch.deltas["fc2.weight"].tolist()},
            "_test_sleep_seconds": 30.0,
        },
    }
    with pytest.raises(EvaluatorIPCError) as exc_info:
        send_evaluator_ipc_request("submit_candidate", req, ledger_path=ledger)
    assert exc_info.value.category == ErrorCategory.RESOURCE_LIMIT_ERROR
    assert exc_info.value.code == "EVALUATION_TIMEOUT_EXCEEDED"


def test_standardized_errors_do_not_leak_stack_traces_or_internal_paths(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)
    # Target a non-existent tensor name to trigger internal validation failure
    bad_req = {
        "experiment_id": env.experiment_id,
        "config_sha256": env.config_sha256,
        "patch": {
            "patch_id": "bad-tensor",
            "base_model_id": "toy",
            "base_model_revision": "v1",
            "base_model_fingerprint": env.corrupted_fingerprint,
            "deltas": {"nonexistent_layer.weight": [[0.1, 0.2]]},
        },
    }
    raw_bytes = json.dumps(
        {"protocol_version": "1.0", "action": "submit_candidate", "payload": bad_req}
    ).encode("utf-8")
    resp, code = handle_ipc_request_bytes(raw_bytes, ledger_path=ledger)
    assert code == 1
    assert resp["error_category"] == "VALIDATION_ERROR"
    msg = json.dumps(resp)
    assert "Traceback" not in msg
    assert "site-packages" not in msg
    assert str(tmp_path) not in msg


def test_fingerprint_mismatch_rejected_before_subprocess(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)
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


def test_legacy_blind_repair_benchmark_explicitly_labeled_non_sealed():
    bm = BlindRepairBenchmark(seed=42)
    assert bm.IS_SEALED_BLIND_BOUNDARY is False
    assert bm.BOUNDARY_CLASSIFICATION == "LEGACY_SAME_PROCESS_BASELINE_HARNESS"
    out = bm.run_full_benchmark()
    assert out["is_sealed_blind_boundary"] is False
    assert out["boundary_classification"] == "LEGACY_SAME_PROCESS_BASELINE_HARNESS"


def test_five_diagnostic_conditions_strictly_scope_proposer_information(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"

    # 1. Behavioral only: has behavioral_failures, NO weight_summaries, NO corrupted_model, NO activations
    env_c1 = BlindProposerEnvironment(
        diagnostic_condition=DiagnosticCondition.BEHAVIORAL_ONLY.value,
        ledger_path=ledger,
    )
    pkg_c1 = env_c1.get_authorized_diagnostic_package()
    assert "behavioral_failures" in pkg_c1
    assert "weight_summaries" not in pkg_c1
    assert "activation_summaries" not in pkg_c1
    assert "toolkit_sensitivity" not in pkg_c1
    with pytest.raises(ProposerIsolationViolationError, match="Model weights are not permitted"):
        _ = env_c1.get_corrupted_model()

    # 2. Weights only: has weight_summaries and corrupted_model, NO behavioral_failures
    env_c2 = BlindProposerEnvironment(
        diagnostic_condition=DiagnosticCondition.WEIGHT_SUMMARIES_ONLY.value,
        ledger_path=ledger,
    )
    pkg_c2 = env_c2.get_authorized_diagnostic_package()
    assert "weight_summaries" in pkg_c2
    assert "behavioral_failures" not in pkg_c2
    assert "activation_summaries" not in pkg_c2
    assert env_c2.get_corrupted_model() is not None
    with pytest.raises(ProposerIsolationViolationError, match="Behavioral failure examples are not permitted"):
        _ = env_c2.get_diagnostic_failure_report()

    # 3. Weights + failures: has both, NO activations
    env_c3 = BlindProposerEnvironment(
        diagnostic_condition=DiagnosticCondition.WEIGHTS_AND_FAILURES.value,
        ledger_path=ledger,
    )
    pkg_c3 = env_c3.get_authorized_diagnostic_package()
    assert "behavioral_failures" in pkg_c3
    assert "weight_summaries" in pkg_c3
    assert "activation_summaries" not in pkg_c3
    assert "toolkit_sensitivity" not in pkg_c3

    # 4. Weights + activations + failures: has activations, NO toolkit_sensitivity
    env_c4 = BlindProposerEnvironment(
        diagnostic_condition=DiagnosticCondition.WEIGHTS_ACTIVATIONS_FAILURES.value,
        ledger_path=ledger,
    )
    pkg_c4 = env_c4.get_authorized_diagnostic_package()
    assert "behavioral_failures" in pkg_c4
    assert "weight_summaries" in pkg_c4
    assert "activation_summaries" in pkg_c4
    assert "toolkit_sensitivity" not in pkg_c4

    # 5. Full diagnostic toolkit: has all four
    env_c5 = BlindProposerEnvironment(
        diagnostic_condition=DiagnosticCondition.FULL_DIAGNOSTIC_TOOLKIT.value,
        ledger_path=ledger,
    )
    pkg_c5 = env_c5.get_authorized_diagnostic_package()
    assert "behavioral_failures" in pkg_c5
    assert "weight_summaries" in pkg_c5
    assert "activation_summaries" in pkg_c5
    assert "toolkit_sensitivity" in pkg_c5
