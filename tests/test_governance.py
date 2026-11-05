import json
import pytest

from weightliftai.experiment_registry import (
    EventChainTamperedError,
    PreregistrationTamperedError,
    append_event,
    build_experiment_manifest,
    create_experiment,
    load_experiment,
    request_hidden_test_access,
    transition_experiment,
    verify_event_chain,
)
from weightliftai.governance import (
    CriticalReviewAudit,
    EvidenceLinkedClaim,
    HumanApprovalGate,
    StageGrader,
)


def sample_manifest():
    return build_experiment_manifest(
        experiment_id="gov-pilot-001",
        hypothesis="Targeted activation steering recovers dialectical intent per Gurnee et al. (2026).",
        model={"model_id": "org/test-model", "revision": "rev123"},
        evaluation={
            "dataset_id": "eval-v1",
            "dataset_sha256": "f" * 64,
            "split": "test",
            "primary_metric": "accuracy",
            "success_criterion": ">= 0.20 improvement",
        },
    )


def test_human_approval_gate_tokens():
    assert HumanApprovalGate.verify_token("INTERVENTION_READY", "human-approval-001") is True
    assert HumanApprovalGate.verify_token("INTERVENTION_READY", "human-intervention_ready-abc123") is True
    assert HumanApprovalGate.verify_token("PROMOTION", "human-promotion-pass123") is True
    assert HumanApprovalGate.verify_token("HIDDEN_TEST_ACCESS", "human-hidden_test_access-gate01") is True

    # Rejection of unauthorized tokens
    assert HumanApprovalGate.verify_token("INTERVENTION_READY", "bot-auto-approval") is False
    assert HumanApprovalGate.verify_token("PROMOTION", "admin") is False
    assert HumanApprovalGate.verify_token("HIDDEN_TEST_ACCESS", "") is False


def test_stage_grader_evaluates_all_six_stages():
    manifest = sample_manifest()
    manifest["results"] = {
        "baseline": {"metrics": {"acc": 0.4}},
        "intervention": {"metrics": {"acc": 0.8}},
    }
    manifest["provenance"]["manifest_sha256"] = "a" * 64

    claim_valid = EvidenceLinkedClaim(
        claim_id="CLM-001",
        statement="Activation steering recovers target task accuracy.",
        claim_level=3,
        evidence_type="activation_intervention",
        source_run_id="gov-pilot-001",
        source_artifact_hashes={"metrics.json": "c" * 64},
        limitations=["Evaluated on synthetic dataset."],
    )

    report = StageGrader.grade(
        manifest=manifest,
        has_citations=True,
        finite_difference_passed=True,
        baselines_present=True,
        negative_controls_present=True,
        claims=[claim_valid],
    )

    assert report["overall_passed"] is True
    assert report["mean_score"] == 1.0
    assert len(report["grades"]) == 6

    # Test failure when claim level exceeds empirical evidence
    claim_overreach = EvidenceLinkedClaim(
        claim_id="CLM-002",
        statement="Autonomous weight edit general repair.",
        claim_level=4,
        evidence_type="diagnostic_run",  # Not a sealed benchmark!
        source_run_id="gov-pilot-001",
        source_artifact_hashes={"metrics.json": "d" * 64},
        limitations=[],
    )

    report_fail = StageGrader.grade(
        manifest=manifest,
        has_citations=True,
        finite_difference_passed=True,
        baselines_present=True,
        negative_controls_present=True,
        claims=[claim_overreach],
    )
    assert report_fail["overall_passed"] is False
    assert report_fail["grades"]["claim_faithfulness"]["passed"] is False


def test_critical_review_audit_detects_leakage_and_unsupported_language():
    manifest = sample_manifest()
    # Leakage test
    diag_ids = {"sample_001", "sample_002", "sample_003"}
    leaked_test_ids = {"sample_003", "sample_004", "sample_005"}

    audit = CriticalReviewAudit.audit_experiment(
        manifest=manifest,
        diagnostic_sample_ids=diag_ids,
        test_sample_ids=leaked_test_ids,
    )
    assert audit["audit_passed"] is False
    assert any("Data Leakage Detected" in v for v in audit["violations"])

    # Unsupported causal language test without intervention results
    manifest["hypothesis"] = "This experiment proves that weight defect causes catastrophic failure."
    audit_causal = CriticalReviewAudit.audit_experiment(
        manifest=manifest,
        diagnostic_sample_ids={"s1"},
        test_sample_ids={"s2"},
    )
    assert audit_causal["audit_passed"] is False
    assert any("Unsupported Causal Language" in v for v in audit_causal["violations"])


def test_event_hash_chain_verification_and_tampering_detection(tmp_path):
    path = create_experiment(tmp_path, sample_manifest())
    transition_experiment(path, "preregistered", actor="researcher", rationale="Freeze.")
    transition_experiment(path, "baseline_complete", actor="runner", rationale="Done.")
    transition_experiment(
        path, "intervention_ready", actor="researcher",
        rationale="Approved.", approval_reference="human-approval-001",
    )

    # 1. Valid chain verification
    status = verify_event_chain(path)
    assert status["valid"] is True
    assert status["event_count"] == 4
    assert status["head_hash"] is not None

    # 2. Tampering test: modify an event line
    events_file = path / "events.jsonl"
    lines = events_file.read_text(encoding="utf-8").splitlines()
    tampered_event = json.loads(lines[1])
    tampered_event["payload"]["actor"] = "malicious_actor"
    lines[1] = json.dumps(tampered_event)
    events_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(EventChainTamperedError, match="invalid event_hash"):
        verify_event_chain(path)


def test_preregistration_immutability_guardrail(tmp_path):
    path = create_experiment(tmp_path, sample_manifest())
    transition_experiment(path, "preregistered", actor="researcher", rationale="Freeze.")

    # Tamper with the evaluation metric in manifest.json after freeze
    manifest_file = path / "manifest.json"
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    data["evaluation"]["success_criterion"] = ">= 0.01 trivial threshold"
    manifest_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Attempting subsequent transition must be blocked
    with pytest.raises(PreregistrationTamperedError, match="modified after protocol freeze"):
        transition_experiment(path, "baseline_complete", actor="runner", rationale="Run baseline.")


def test_hidden_test_access_gate(tmp_path):
    path = create_experiment(tmp_path, sample_manifest())
    transition_experiment(path, "preregistered", actor="researcher", rationale="Freeze.")
    transition_experiment(path, "baseline_complete", actor="runner", rationale="Baseline.")
    transition_experiment(
        path, "intervention_ready", actor="researcher",
        rationale="Approved.", approval_reference="human-approval-001",
    )
    transition_experiment(path, "intervention_complete", actor="runner", rationale="Patch done.")

    # Valid human access
    event = request_hidden_test_access(
        path, actor="evaluator", approval_reference="human-hidden_test_access-eval456"
    )
    assert event["event_type"] == "hidden_test_accessed"

    # Reject non-human or invalid token
    with pytest.raises(ValueError, match="Invalid human approval reference"):
        request_hidden_test_access(
            path, actor="evaluator", approval_reference="bot-auto-bypass"
        )
