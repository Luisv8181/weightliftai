"""Append-only, local-first registry for controlled model experiments.

This module records experiment plans and lifecycle events. It deliberately does not
load models, execute generated code, or modify checkpoints; those belong in a
separate, explicitly approved execution layer.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
STATES = (
    "draft",
    "preregistered",
    "baseline_complete",
    "intervention_ready",
    "intervention_complete",
    "evaluated",
    "independently_reviewed",
    "accepted",
    "rejected",
    "inconclusive",
)
TERMINAL_STATES = {"accepted", "rejected", "inconclusive"}
TRANSITIONS = {
    "draft": {"preregistered"},
    "preregistered": {"baseline_complete"},
    "baseline_complete": {"intervention_ready"},
    "intervention_ready": {"intervention_complete"},
    "intervention_complete": {"evaluated"},
    "evaluated": {"independently_reviewed"},
    "independently_reviewed": {"accepted", "rejected", "inconclusive"},
    "accepted": set(),
    "rejected": set(),
    "inconclusive": set(),
}
_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _safe_id(value: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise ValueError("experiment_id must be 1-80 safe characters (letters, digits, . _ -)")
    return value


def build_experiment_manifest(
    *,
    hypothesis: str,
    model: dict[str, Any],
    evaluation: dict[str, Any],
    intervention: dict[str, Any] | None = None,
    experiment_id: str | None = None,
    parent_experiment_id: str | None = None,
    agent: dict[str, Any] | None = None,
    resource_budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a preregistration-ready manifest; callers must fill exact provenance."""
    if not hypothesis.strip():
        raise ValueError("hypothesis must not be empty")
    if not model.get("model_id") or not model.get("revision"):
        raise ValueError("model.model_id and model.revision are required")
    if not evaluation.get("dataset_id") or not evaluation.get("split"):
        raise ValueError("evaluation.dataset_id and evaluation.split are required")
    primary = evaluation.get("primary_metric")
    if not primary or not evaluation.get("success_criterion"):
        raise ValueError("evaluation.primary_metric and success_criterion are required")
    run_id = _safe_id(experiment_id or f"exp-{uuid.uuid4().hex[:12]}")
    if parent_experiment_id:
        _safe_id(parent_experiment_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": run_id,
        "parent_experiment_id": parent_experiment_id,
        "created_at_utc": utc_now(),
        "state": "draft",
        "hypothesis": hypothesis.strip(),
        "model": model,
        "evaluation": evaluation,
        "intervention": intervention or {"type": "none", "description": "Baseline only"},
        "agent": agent or {"role": "human", "name": "unspecified"},
        "resource_budget": resource_budget or {},
        "provenance": {
            "code_commit": None,
            "environment_lock_hash": None,
            "manifest_sha256": None,
        },
        "results": {},
        "limitations": [],
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version", "experiment_id", "state", "hypothesis",
        "model", "evaluation", "intervention", "provenance", "results",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"manifest missing required keys: {sorted(missing)}")
    _safe_id(manifest["experiment_id"])
    if manifest["state"] not in STATES:
        raise ValueError(f"unknown experiment state: {manifest['state']}")
    if not manifest["hypothesis"].strip():
        raise ValueError("hypothesis must not be empty")
    if not manifest["model"].get("model_id") or not manifest["model"].get("revision"):
        raise ValueError("model identity and immutable revision are required")
    if not manifest["evaluation"].get("dataset_id") or not manifest["evaluation"].get("split"):
        raise ValueError("dataset identity and split are required")
    if not manifest["evaluation"].get("primary_metric"):
        raise ValueError("a preregistered primary_metric is required")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_experiment(root: str | Path, manifest: dict[str, Any]) -> Path:
    """Persist a new experiment without overwriting an existing run."""
    validate_manifest(manifest)
    if manifest["state"] != "draft":
        raise ValueError("new experiments must start in draft state")
    destination = Path(root).expanduser() / manifest["experiment_id"]
    destination.mkdir(parents=True, exist_ok=False)
    stored = dict(manifest)
    stored["provenance"] = dict(stored["provenance"])
    stored["provenance"]["manifest_sha256"] = sha256_json(
        {k: v for k, v in stored.items() if k != "provenance"}
    )
    _write_json_atomic(destination / "manifest.json", stored)
    append_event(destination, "experiment_created", {"state": "draft"})
    return destination


def load_experiment(path: str | Path) -> dict[str, Any]:
    manifest = json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def append_event(experiment_dir: str | Path, event_type: str, payload: dict[str, Any]) -> None:
    """Append an event; existing history is never rewritten."""
    if not event_type.strip():
        raise ValueError("event_type must not be empty")
    directory = Path(experiment_dir)
    directory.mkdir(parents=True, exist_ok=True)
    event = {"timestamp_utc": utc_now(), "event_type": event_type, "payload": payload}
    with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(event) + "\n")


def transition_experiment(
    experiment_dir: str | Path,
    new_state: str,
    *,
    actor: str,
    rationale: str,
    approval_reference: str | None = None,
) -> dict[str, Any]:
    """Apply a permitted lifecycle transition and record who authorized it."""
    directory = Path(experiment_dir)
    manifest = load_experiment(directory)
    current = manifest["state"]
    if new_state not in TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid transition: {current} -> {new_state}")
    if not actor.strip() or not rationale.strip():
        raise ValueError("actor and rationale are required")
    if new_state in {"intervention_ready", "accepted"} and not approval_reference:
        raise ValueError(f"approval_reference is required before {new_state}")
    manifest["state"] = new_state
    manifest["updated_at_utc"] = utc_now()
    _write_json_atomic(directory / "manifest.json", manifest)
    append_event(directory, "state_transition", {
        "from": current,
        "to": new_state,
        "actor": actor,
        "rationale": rationale,
        "approval_reference": approval_reference,
    })
    return manifest


def record_results(
    experiment_dir: str | Path,
    results: dict[str, Any],
    *,
    actor: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Attach raw-result references and metrics; never accept an unprovenanced result."""
    if not source_hashes:
        raise ValueError("source_hashes are required to identify raw result artifacts")
    if not actor.strip():
        raise ValueError("actor is required")
    directory = Path(experiment_dir)
    manifest = load_experiment(directory)
    if manifest["state"] not in {"baseline_complete", "intervention_complete"}:
        raise ValueError("results may only be recorded after a baseline or intervention run")
    entry = {
        "recorded_at_utc": utc_now(),
        "actor": actor,
        "metrics": results,
        "source_hashes": source_hashes,
    }
    key = "baseline" if manifest["state"] == "baseline_complete" else "intervention"
    if key in manifest["results"]:
        raise ValueError(f"{key} results already exist; create a linked replication instead")
    manifest["results"][key] = entry
    _write_json_atomic(directory / "manifest.json", manifest)
    append_event(directory, f"{key}_results_recorded", {
        "actor": actor, "source_hashes": source_hashes, "metrics_sha256": sha256_json(results)
    })
    return manifest


def read_events(experiment_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(experiment_dir) / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
