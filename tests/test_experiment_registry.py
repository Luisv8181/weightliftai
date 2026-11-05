import json

import pytest

from weightliftai.experiment_registry import (
    append_event,
    build_experiment_manifest,
    create_experiment,
    load_experiment,
    read_events,
    record_results,
    transition_experiment,
)


def sample_manifest():
    return build_experiment_manifest(
        experiment_id="pilot-001",
        hypothesis="Activation steering improves contextual intent preservation.",
        model={"model_id": "org/model", "revision": "abc123", "tokenizer_revision": "abc123"},
        evaluation={
            "dataset_id": "context-probes-v1",
            "dataset_sha256": "a" * 64,
            "split": "test",
            "primary_metric": "intent_preservation",
            "success_criterion": ">= 0.05 absolute improvement",
            "regression_gates": {"general_spanish_fluency": "no more than 0.02 decline"},
        },
    )


def test_create_experiment_is_immutable_and_records_event(tmp_path):
    path = create_experiment(tmp_path, sample_manifest())
    loaded = load_experiment(path)
    assert loaded["state"] == "draft"
    assert len(loaded["provenance"]["manifest_sha256"]) == 64
    assert [event["event_type"] for event in read_events(path)] == ["experiment_created"]
    with pytest.raises(FileExistsError):
        create_experiment(tmp_path, sample_manifest())


def test_manifest_requires_fixed_model_and_evaluation(tmp_path):
    manifest = sample_manifest()
    manifest["model"]["revision"] = ""
    with pytest.raises(ValueError, match="immutable revision"):
        create_experiment(tmp_path, manifest)


def test_lifecycle_requires_approval_for_intervention_and_acceptance(tmp_path):
    path = create_experiment(tmp_path, sample_manifest())
    transition_experiment(path, "preregistered", actor="researcher", rationale="Freeze protocol.")
    transition_experiment(path, "baseline_complete", actor="runner", rationale="Baseline finished.")
    with pytest.raises(ValueError, match="approval_reference"):
        transition_experiment(path, "intervention_ready", actor="researcher", rationale="Proceed.")
    transition_experiment(
        path, "intervention_ready", actor="researcher",
        rationale="Reviewed bounded intervention.", approval_reference="human-approval-001",
    )
    transition_experiment(path, "intervention_complete", actor="runner", rationale="Patch run finished.")
    transition_experiment(path, "evaluated", actor="evaluator", rationale="Regression suite completed.")
    transition_experiment(path, "independently_reviewed", actor="reviewer", rationale="Independent review.")
    with pytest.raises(ValueError, match="approval_reference"):
        transition_experiment(path, "accepted", actor="reviewer", rationale="Promote.")
    transition_experiment(
        path, "accepted", actor="human", rationale="All preregistered gates passed.",
        approval_reference="human-promotion-001",
    )
    assert load_experiment(path)["state"] == "accepted"


def test_invalid_transition_is_rejected(tmp_path):
    path = create_experiment(tmp_path, sample_manifest())
    with pytest.raises(ValueError, match="invalid transition"):
        transition_experiment(path, "accepted", actor="human", rationale="Skip the science.")


def test_results_require_provenance_and_cannot_be_overwritten(tmp_path):
    path = create_experiment(tmp_path, sample_manifest())
    transition_experiment(path, "preregistered", actor="researcher", rationale="Freeze.")
    transition_experiment(path, "baseline_complete", actor="runner", rationale="Baseline complete.")
    with pytest.raises(ValueError, match="source_hashes"):
        record_results(path, {"score": 0.7}, actor="runner", source_hashes={})
    record_results(
        path, {"score": 0.7}, actor="runner",
        source_hashes={"outputs.jsonl": "b" * 64},
    )
    with pytest.raises(ValueError, match="already exist"):
        record_results(
            path, {"score": 0.8}, actor="runner",
            source_hashes={"outputs.jsonl": "c" * 64},
        )
    saved = json.loads((path / "manifest.json").read_text())
    assert saved["results"]["baseline"]["metrics"]["score"] == 0.7


def test_event_log_is_append_only(tmp_path):
    path = create_experiment(tmp_path, sample_manifest())
    append_event(path, "agent_note", {"note": "A failed attempt is still recorded."})
    assert len(read_events(path)) == 2
