"""Evaluator-Authoritative Experiment Registry, Private Vault, and Execution Engine.

Implements:
- Prompt 1: Evaluator-side experiment registration, query authorization, persistent SQLite ledger
  across instances/restarts/concurrency, consume-before-validation attempt policy, and Ed25519
  signed aggregate receipts.
- Prompt 2: Private evaluator-controlled randomness for benchmark instances and authorized
  researcher audit/replay records.
- Prompt 4: Pre-allocation resource limits, execution timeouts, and CPU thread caps.
- Prompt 5: Standardized error categories, reduced diagnostic side channels, and separation
  between proposer-visible aggregate feedback and researcher-only audit records.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
import tempfile
import time
from typing import Any

import numpy as np
import torch

from weightliftai.weight_programming.defect_families import (
    DefectFamily,
    DiagnosticCondition,
    deserialize_model_state_dict,
    generate_private_benchmark_instance,
)
from weightliftai.weight_programming.evaluator import IndependentEvaluator
from weightliftai.weight_programming.inspector import (
    compute_model_fingerprint,
    validate_patch_against_budget,
)
from weightliftai.weight_programming.ipc_protocol import (
    DEFAULT_RESOURCE_LIMITS,
    EVALUATOR_VERSION,
    ErrorCategory,
    EvaluatorIPCError,
    EvaluatorResourceLimits,
    compute_config_sha256,
    generate_ed25519_keypair_hex,
    prevalidate_and_deserialize_deltas,
    sign_evaluation_receipt,
    verify_signed_evaluation_result,
)
from weightliftai.weight_programming.schemas import (
    AllowlistViolationError,
    BaseModelMismatchError,
    BudgetViolationError,
    PatchBudget,
    WeightPatch,
)
from weightliftai.weight_programming.toy_defect import mse_evaluator


ATTEMPT_CONSUMPTION_POLICY = "CONSUME_BEFORE_VALIDATION"
DEFAULT_EVALUATOR_ALLOWED_SUBMISSIONS = 1
DEFAULT_EXPIRATION_SECONDS = 3600.0

_DEFAULT_VAULT_DIR = Path(tempfile.gettempdir()) / "weightliftai_evaluator_vault_v2"


def get_default_ledger_path() -> Path:
    """Return the active evaluator SQLite ledger path (configurable via env var for isolated tests)."""
    env_path = os.environ.get("WEIGHTLIFTAI_EVALUATOR_LEDGER_PATH")
    if env_path:
        p = Path(env_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    _DEFAULT_VAULT_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_VAULT_DIR / "evaluator_authority_ledger.sqlite3"


class EvaluatorRegistryAuthority:
    """Evaluator-controlled persistent ledger and private instance vault."""

    def __init__(
        self,
        ledger_path: Path | str | None = None,
        resource_limits: EvaluatorResourceLimits = DEFAULT_RESOURCE_LIMITS,
    ):
        self.ledger_path = Path(ledger_path) if ledger_path is not None else get_default_ledger_path()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.resource_limits = resource_limits
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.ledger_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=15000;")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluator_keys (
                    key_id TEXT PRIMARY KEY,
                    private_key_hex TEXT NOT NULL,
                    public_key_hex TEXT NOT NULL,
                    researcher_audit_token TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    benchmark_instance_id TEXT NOT NULL,
                    defect_family TEXT NOT NULL,
                    diagnostic_condition TEXT NOT NULL,
                    allowed_submissions INTEGER NOT NULL,
                    attempts_used INTEGER NOT NULL DEFAULT 0,
                    attempt_policy TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    corrupted_fingerprint TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    config_sha256 TEXT NOT NULL,
                    private_vault_json TEXT NOT NULL,
                    proposer_bundle_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS submission_audit_log (
                    submission_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    attempt_index INTEGER NOT NULL,
                    candidate_patch_id TEXT,
                    candidate_patch_sha256 TEXT,
                    status TEXT NOT NULL,
                    error_category TEXT,
                    error_code TEXT,
                    proposer_response_json TEXT NOT NULL,
                    researcher_audit_json TEXT NOT NULL,
                    submitted_at REAL NOT NULL
                );
                """
            )

            row = conn.execute("SELECT * FROM evaluator_keys WHERE key_id = 'primary'").fetchone()
            if row is None:
                priv_hex, pub_hex = generate_ed25519_keypair_hex()
                audit_token = f"audit_{secrets.token_hex(24)}"
                conn.execute(
                    """
                    INSERT INTO evaluator_keys (key_id, private_key_hex, public_key_hex, researcher_audit_token, created_at)
                    VALUES ('primary', ?, ?, ?, ?)
                    """,
                    (priv_hex, pub_hex, audit_token, time.time()),
                )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def get_public_key_hex(self) -> str:
        conn = self._connect()
        try:
            row = conn.execute("SELECT public_key_hex FROM evaluator_keys WHERE key_id = 'primary'").fetchone()
            return str(row["public_key_hex"])
        finally:
            conn.close()

    def get_researcher_audit_token(self) -> str:
        """Retrieve the researcher audit token (for authorized test/researcher audit verification)."""
        conn = self._connect()
        try:
            row = conn.execute("SELECT researcher_audit_token FROM evaluator_keys WHERE key_id = 'primary'").fetchone()
            return str(row["researcher_audit_token"])
        finally:
            conn.close()

    def _get_signing_keys(self, conn: sqlite3.Connection) -> tuple[str, str]:
        row = conn.execute(
            "SELECT private_key_hex, public_key_hex FROM evaluator_keys WHERE key_id = 'primary'"
        ).fetchone()
        return str(row["private_key_hex"]), str(row["public_key_hex"])

    def register_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Authoritatively register a new blind experiment with private evaluator randomness."""
        # Forbid proposer from choosing its own experiment_id or injecting private seeds/clean weights
        forbidden_caller_keys = {
            "experiment_id",
            "private_seed",
            "clean_state_dict",
            "defect_mask",
            "allowed_submissions",
            "max_queries",
        }
        present_forbidden = forbidden_caller_keys.intersection(payload.keys())
        if present_forbidden:
            raise EvaluatorIPCError(
                ErrorCategory.AUTH_ERROR,
                "PROPOSER_CONFIGURATION_OVERRIDE_FORBIDDEN",
                f"Proposer may not specify evaluator-controlled registration fields: {sorted(present_forbidden)}.",
            )

        defect_family = str(payload.get("defect_family", DefectFamily.FC2_ROW0_ZERO_MASK.value))
        diagnostic_condition = str(
            payload.get("diagnostic_condition", DiagnosticCondition.FULL_DIAGNOSTIC_TOOLKIT.value)
        )
        ttl_seconds = float(payload.get("ttl_seconds", DEFAULT_EXPIRATION_SECONDS))
        if ttl_seconds <= 0.0 or ttl_seconds > 86400.0:
            raise EvaluatorIPCError(
                ErrorCategory.VALIDATION_ERROR,
                "INVALID_TTL_POLICY",
                "Experiment TTL must be in (0, 86400] seconds.",
            )

        # Generate unpredictable experiment ID and private evaluator seed (Prompt 1 & Prompt 2)
        experiment_id = f"exp_{secrets.token_hex(16)}"
        benchmark_instance_id = f"inst_{secrets.token_hex(12)}"
        private_seed = secrets.randbits(31)
        if private_seed == 0:
            private_seed = 1

        try:
            private_vault, proposer_bundle = generate_private_benchmark_instance(
                private_seed=private_seed,
                defect_family=defect_family,
                diagnostic_condition=diagnostic_condition,
            )
        except ValueError as exc:
            raise EvaluatorIPCError(
                ErrorCategory.VALIDATION_ERROR,
                "INVALID_EXPERIMENT_PARAMETERS",
                str(exc),
            ) from exc

        now = time.time()
        expires_at = now + ttl_seconds
        allowed_submissions = DEFAULT_EVALUATOR_ALLOWED_SUBMISSIONS
        corrupted_fingerprint = private_vault["corrupted_fingerprint"]

        immutable_config = {
            "experiment_id": experiment_id,
            "benchmark_instance_id": benchmark_instance_id,
            "defect_family": defect_family,
            "diagnostic_condition": diagnostic_condition,
            "allowed_submissions": allowed_submissions,
            "attempt_policy": ATTEMPT_CONSUMPTION_POLICY,
            "corrupted_fingerprint": corrupted_fingerprint,
            "evaluator_version": EVALUATOR_VERSION,
        }
        config_sha256 = compute_config_sha256(immutable_config)

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            _, pub_key_hex = self._get_signing_keys(conn)
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, benchmark_instance_id, defect_family, diagnostic_condition,
                    allowed_submissions, attempts_used, attempt_policy, status,
                    created_at, expires_at, corrupted_fingerprint,
                    config_json, config_sha256, private_vault_json, proposer_bundle_json
                ) VALUES (?, ?, ?, ?, ?, 0, ?, 'ACTIVE', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    benchmark_instance_id,
                    defect_family,
                    diagnostic_condition,
                    allowed_submissions,
                    ATTEMPT_CONSUMPTION_POLICY,
                    now,
                    expires_at,
                    corrupted_fingerprint,
                    json.dumps(immutable_config, sort_keys=True),
                    config_sha256,
                    json.dumps(private_vault),
                    json.dumps(proposer_bundle),
                ),
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

        return {
            "experiment_id": experiment_id,
            "benchmark_instance_id": benchmark_instance_id,
            "defect_family": defect_family,
            "diagnostic_condition": diagnostic_condition,
            "allowed_submissions": allowed_submissions,
            "attempt_policy": ATTEMPT_CONSUMPTION_POLICY,
            "expires_at": expires_at,
            "config_sha256": config_sha256,
            "evaluator_public_key": pub_key_hex,
            "proposer_bundle": proposer_bundle,
        }

    def submit_candidate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Authorize, consume attempt, validate, and evaluate a candidate patch."""
        start_wall = time.perf_counter()
        torch.set_num_threads(self.resource_limits.max_cpu_threads)

        experiment_id = payload.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id.startswith("exp_"):
            raise EvaluatorIPCError(
                ErrorCategory.AUTH_ERROR,
                "INVALID_OR_MISSING_EXPERIMENT_ID",
                "A valid evaluator-generated experiment_id is required for every submission.",
            )

        declared_config_sha256 = payload.get("config_sha256")
        if not isinstance(declared_config_sha256, str) or len(declared_config_sha256) != 64:
            raise EvaluatorIPCError(
                ErrorCategory.AUTH_ERROR,
                "MISSING_CONFIG_BINDING",
                "Submission must bind to the registered experiment config_sha256.",
            )

        # Step 1: Atomic authorization check + CONSUME_BEFORE_VALIDATION in SQLite
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()

            if row is None:
                conn.execute("ROLLBACK;")
                raise EvaluatorIPCError(
                    ErrorCategory.AUTH_ERROR,
                    "UNKNOWN_EXPERIMENT_ID",
                    "Unknown or unregistered experiment_id.",
                )

            now = time.time()
            if now > float(row["expires_at"]) or row["status"] == "EXPIRED":
                conn.execute(
                    "UPDATE experiments SET status = 'EXPIRED' WHERE experiment_id = ?",
                    (experiment_id,),
                )
                conn.execute("COMMIT;")
                raise EvaluatorIPCError(
                    ErrorCategory.AUTH_ERROR,
                    "EXPERIMENT_EXPIRED",
                    "Experiment registration has expired.",
                )

            if str(row["config_sha256"]) != declared_config_sha256:
                conn.execute("ROLLBACK;")
                raise EvaluatorIPCError(
                    ErrorCategory.AUTH_ERROR,
                    "EXPERIMENT_CONFIG_MISMATCH",
                    "Submission config_sha256 does not match the registered immutable configuration.",
                )

            # Reject caller attempts to alter diagnostic_condition, defect_family, or allowed_submissions at submission time
            for immutable_field in ("defect_family", "diagnostic_condition", "allowed_submissions"):
                if immutable_field in payload and payload[immutable_field] != row[immutable_field]:
                    conn.execute("ROLLBACK;")
                    raise EvaluatorIPCError(
                        ErrorCategory.AUTH_ERROR,
                        "IMMUTABLE_REGISTRATION_ALTERED",
                        "Submission attempted to alter immutable registration parameters.",
                    )

            attempts_used = int(row["attempts_used"])
            allowed_submissions = int(row["allowed_submissions"])
            if row["status"] != "ACTIVE" or attempts_used >= allowed_submissions:
                conn.execute("ROLLBACK;")
                raise EvaluatorIPCError(
                    ErrorCategory.AUTH_ERROR,
                    "QUERY_BUDGET_EXCEEDED",
                    f"Evaluation query budget exceeded ({attempts_used}/{allowed_submissions}).",
                )

            # CONSUME_BEFORE_VALIDATION: increment attempts_used immediately
            new_attempts = attempts_used + 1
            new_status = "COMPLETED" if new_attempts >= allowed_submissions else "ACTIVE"
            conn.execute(
                """
                UPDATE experiments
                SET attempts_used = ?, status = ?
                WHERE experiment_id = ?
                """,
                (new_attempts, new_status, experiment_id),
            )
            priv_key_hex, pub_key_hex = self._get_signing_keys(conn)
            private_vault = json.loads(row["private_vault_json"])
            config_sha256 = str(row["config_sha256"])
            corrupted_fingerprint = str(row["corrupted_fingerprint"])
            conn.execute("COMMIT;")
        except EvaluatorIPCError:
            raise
        except Exception as exc:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise EvaluatorIPCError(
                ErrorCategory.INTERNAL_EVALUATOR_ERROR,
                "LEDGER_TRANSACTION_FAILURE",
                "Evaluator failed to process authorization transaction.",
                internal_detail=str(exc),
            ) from exc
        finally:
            conn.close()

        # Step 2: Validate candidate patch schema, pre-allocation limits, and hash binding
        try:
            raw_patch = payload.get("patch")
            if not isinstance(raw_patch, dict):
                raise EvaluatorIPCError(
                    ErrorCategory.PROTOCOL_ERROR,
                    "MISSING_PATCH_OBJECT",
                    "Submission payload must contain a 'patch' dictionary.",
                )

            for req_key in ("patch_id", "base_model_id", "base_model_revision", "base_model_fingerprint", "deltas"):
                if req_key not in raw_patch:
                    raise EvaluatorIPCError(
                        ErrorCategory.PROTOCOL_ERROR,
                        "MISSING_PATCH_FIELD",
                        f"Candidate patch is missing required field '{req_key}'.",
                    )

            deltas = prevalidate_and_deserialize_deltas(
                raw_patch.get("deltas"),
                declared_shapes=raw_patch.get("declared_shapes"),
                declared_dtypes=raw_patch.get("declared_dtypes"),
                limits=self.resource_limits,
            )

            patch = WeightPatch(
                patch_id=str(raw_patch["patch_id"]),
                base_model_id=str(raw_patch["base_model_id"]),
                base_model_revision=str(raw_patch["base_model_revision"]),
                base_model_fingerprint=str(raw_patch["base_model_fingerprint"]),
                deltas=deltas,
                hypothesis=str(raw_patch.get("hypothesis", "")),
                proposed_by=str(raw_patch.get("proposed_by", "unspecified")),
                rationale=str(raw_patch.get("rationale", "")),
            )

            actual_patch_sha256 = patch.sha256()
            declared_patch_sha256 = payload.get("candidate_patch_sha256")
            if declared_patch_sha256 is not None and str(declared_patch_sha256) != actual_patch_sha256:
                raise EvaluatorIPCError(
                    ErrorCategory.PROTOCOL_ERROR,
                    "CANDIDATE_HASH_MISMATCH",
                    "Submitted patch content does not match declared candidate_patch_sha256.",
                )

            # Check simulated delay hook for timeout testing if requested within test harness
            sim_delay = float(raw_patch.get("_test_sleep_seconds", 0.0))
            if sim_delay > 0.0:
                if sim_delay > self.resource_limits.max_eval_duration_seconds:
                    raise EvaluatorIPCError(
                        ErrorCategory.RESOURCE_LIMIT_ERROR,
                        "EVALUATION_TIMEOUT_EXCEEDED",
                        "Evaluation exceeded maximum permitted wall-clock duration.",
                    )
                time.sleep(sim_delay)

            # Reconstruct private model and splits inside evaluator memory
            corrupted_model = deserialize_model_state_dict(private_vault["corrupted_state_dict"])
            actual_corrupted_fp = compute_model_fingerprint(corrupted_model)
            if patch.base_model_fingerprint != actual_corrupted_fp or actual_corrupted_fp != corrupted_fingerprint:
                raise EvaluatorIPCError(
                    ErrorCategory.VALIDATION_ERROR,
                    "BASE_FINGERPRINT_MISMATCH",
                    "Candidate patch base_model_fingerprint does not match registered corrupted model.",
                )

            b_cfg = private_vault["budget"]
            budget = PatchBudget(
                max_changed_tensors=int(b_cfg["max_changed_tensors"]),
                max_l2_norm=float(b_cfg["max_l2_norm"]),
                max_relative_norm=float(b_cfg["max_relative_norm"]),
                allowed_tensor_names=list(b_cfg["allowed_tensor_names"]),
            )

            try:
                budget_report = validate_patch_against_budget(corrupted_model, patch, budget)
            except (BaseModelMismatchError, AllowlistViolationError, BudgetViolationError, ValueError) as val_exc:
                raise EvaluatorIPCError(
                    ErrorCategory.VALIDATION_ERROR,
                    "PATCH_BUDGET_OR_INVARIANT_VIOLATION",
                    "Candidate patch failed budget, allowlist, or structural validation.",
                    internal_detail=str(val_exc),
                ) from val_exc

            splits = {}
            for k, v in private_vault["splits"].items():
                splits[k] = {
                    "x": torch.tensor(v["x"], dtype=torch.float32),
                    "y": torch.tensor(v["y"], dtype=torch.float32),
                    "target_col": v.get("target_col"),
                }

            evaluator = IndependentEvaluator(
                eval_fn=mse_evaluator,
                target_in_domain=splits["in_domain"],
                target_held_out=splits["held_out"],
                distribution_shift=splits["distribution_shift"],
                regression_suite=splits["regression"],
                min_target_gain=0.10,
                max_regression_tolerance=0.02,
                higher_is_better=True,
            )

            eval_result = evaluator.evaluate_patch(corrupted_model, patch, budget=budget)
            elapsed = time.perf_counter() - start_wall
            if elapsed > self.resource_limits.max_eval_duration_seconds:
                raise EvaluatorIPCError(
                    ErrorCategory.RESOURCE_LIMIT_ERROR,
                    "EVALUATION_TIMEOUT_EXCEEDED",
                    "Evaluation exceeded maximum permitted wall-clock duration.",
                )

            # Separate coarse rounded proposer-visible metrics from full researcher audit metrics (Prompt 5)
            coarse_aggregate_scores = {
                "held_out_mse": round(-float(eval_result["metrics"]["patched"]["held_out"]), 4),
                "ood_mse": round(-float(eval_result["metrics"]["patched"]["distribution_shift"]), 4),
                "regression_mse": round(-float(eval_result["metrics"]["patched"]["regression_suite"]), 4),
            }
            coarse_gains = {
                k: round(float(v), 4) for k, v in eval_result["gains"].items()
            }

            issued_at = round(time.time(), 3)
            unsigned_receipt = {
                "experiment_id": experiment_id,
                "candidate_patch_id": patch.patch_id,
                "candidate_patch_sha256": actual_patch_sha256,
                "corrupted_fingerprint": corrupted_fingerprint,
                "evaluator_version": EVALUATOR_VERSION,
                "config_sha256": config_sha256,
                "passed_gates": bool(eval_result["passed_gates"]),
                "recommendation": str(eval_result["recommendation"]),
                "aggregate_scores": coarse_aggregate_scores,
                "issued_at": issued_at,
                "evaluator_public_key": pub_key_hex,
            }
            signed_receipt = sign_evaluation_receipt(unsigned_receipt, priv_key_hex)

            proposer_response = {
                "success": True,
                "experiment_id": experiment_id,
                "attempt_index": new_attempts,
                "attempt_policy": ATTEMPT_CONSUMPTION_POLICY,
                "patch_id": patch.patch_id,
                "patch_sha256": actual_patch_sha256,
                "passed_gates": bool(eval_result["passed_gates"]),
                "recommendation": str(eval_result["recommendation"]),
                "gains": coarse_gains,
                "gate_thresholds": eval_result["gate_thresholds"],
                "aggregate_scores": coarse_aggregate_scores,
                "signed_receipt": signed_receipt,
            }

            researcher_audit = {
                "experiment_id": experiment_id,
                "attempt_index": new_attempts,
                "elapsed_seconds": elapsed,
                "private_seed": private_vault["private_seed"],
                "defect_family": private_vault["defect_family"],
                "diagnostic_condition": private_vault["diagnostic_condition"],
                "full_precision_eval_result": eval_result,
                "budget_report": budget_report,
                "signed_receipt": signed_receipt,
            }

            self._record_submission_audit(
                experiment_id=experiment_id,
                attempt_index=new_attempts,
                candidate_patch_id=patch.patch_id,
                candidate_patch_sha256=actual_patch_sha256,
                status="SUCCEEDED",
                error_category=None,
                error_code=None,
                proposer_response=proposer_response,
                researcher_audit=researcher_audit,
            )
            return proposer_response

        except EvaluatorIPCError as ipc_err:
            self._record_submission_audit(
                experiment_id=experiment_id,
                attempt_index=new_attempts,
                candidate_patch_id=str(payload.get("patch", {}).get("patch_id", "")) if isinstance(payload.get("patch"), dict) else None,
                candidate_patch_sha256=None,
                status="FAILED",
                error_category=ipc_err.category.value,
                error_code=ipc_err.code,
                proposer_response={"error_category": ipc_err.category.value, "error_code": ipc_err.code},
                researcher_audit={"internal_detail": ipc_err.internal_detail},
            )
            raise

    def _record_submission_audit(
        self,
        experiment_id: str,
        attempt_index: int,
        candidate_patch_id: str | None,
        candidate_patch_sha256: str | None,
        status: str,
        error_category: str | None,
        error_code: str | None,
        proposer_response: dict[str, Any],
        researcher_audit: dict[str, Any],
    ) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT INTO submission_audit_log (
                    experiment_id, attempt_index, candidate_patch_id, candidate_patch_sha256,
                    status, error_category, error_code, proposer_response_json, researcher_audit_json, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    attempt_index,
                    candidate_patch_id,
                    candidate_patch_sha256,
                    status,
                    error_category,
                    error_code,
                    json.dumps(proposer_response),
                    json.dumps(researcher_audit),
                    time.time(),
                ),
            )
            conn.execute("COMMIT;")
        finally:
            conn.close()

    def verify_receipt(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Verify both Ed25519 cryptographic signature and ledger existence of a result."""
        result_obj = payload.get("evaluation_result")
        if not isinstance(result_obj, dict):
            return {"verified": False, "reason": "INVALID_RESULT_OBJECT"}

        pub_key_hex = self.get_public_key_hex()
        if not verify_signed_evaluation_result(result_obj, public_key_hex=pub_key_hex):
            return {"verified": False, "reason": "SIGNATURE_OR_BINDING_INVALID"}

        receipt = result_obj["signed_receipt"]
        exp_id = receipt["experiment_id"]
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT status, candidate_patch_sha256, proposer_response_json
                FROM submission_audit_log
                WHERE experiment_id = ? AND status = 'SUCCEEDED'
                ORDER BY submission_seq DESC LIMIT 1
                """,
                (exp_id,),
            ).fetchone()
            if row is None:
                return {"verified": False, "reason": "NO_MATCHING_LEDGER_RECORD"}
            if str(row["candidate_patch_sha256"]) != str(receipt["candidate_patch_sha256"]):
                return {"verified": False, "reason": "LEDGER_PATCH_HASH_MISMATCH"}
            return {"verified": True, "experiment_id": exp_id, "evaluator_public_key": pub_key_hex}
        finally:
            conn.close()

    def researcher_audit_replay(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Authorized researcher-only endpoint to inspect full audit records and replay provenance."""
        provided_token = payload.get("researcher_audit_token")
        expected_token = self.get_researcher_audit_token()
        if not isinstance(provided_token, str) or not secrets.compare_digest(provided_token, expected_token):
            raise EvaluatorIPCError(
                ErrorCategory.AUTH_ERROR,
                "UNAUTHORIZED_RESEARCHER_AUDIT_ACCESS",
                "Valid researcher_audit_token is required to access private audit records.",
            )

        exp_id = payload.get("experiment_id")
        conn = self._connect()
        try:
            exp_row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()
            if exp_row is None:
                raise EvaluatorIPCError(
                    ErrorCategory.AUTH_ERROR,
                    "UNKNOWN_EXPERIMENT_ID",
                    "Requested experiment_id not found in audit ledger.",
                )
            sub_rows = conn.execute(
                "SELECT * FROM submission_audit_log WHERE experiment_id = ? ORDER BY submission_seq ASC",
                (exp_id,),
            ).fetchall()

            private_vault = json.loads(exp_row["private_vault_json"])
            return {
                "experiment_id": exp_id,
                "benchmark_instance_id": exp_row["benchmark_instance_id"],
                "defect_family": exp_row["defect_family"],
                "diagnostic_condition": exp_row["diagnostic_condition"],
                "status": exp_row["status"],
                "attempts_used": exp_row["attempts_used"],
                "allowed_submissions": exp_row["allowed_submissions"],
                "config_sha256": exp_row["config_sha256"],
                "private_seed": private_vault["private_seed"],
                "corrupted_fingerprint": exp_row["corrupted_fingerprint"],
                "submissions": [
                    {
                        "attempt_index": r["attempt_index"],
                        "status": r["status"],
                        "error_category": r["error_category"],
                        "error_code": r["error_code"],
                        "researcher_audit": json.loads(r["researcher_audit_json"]),
                    }
                    for r in sub_rows
                ],
            }
        finally:
            conn.close()
