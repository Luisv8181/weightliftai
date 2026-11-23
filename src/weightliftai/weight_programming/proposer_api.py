"""Proposer API Environment for Blind Weight-Repair Benchmark.

Provides a clean, narrow capability interface to the candidate repair agent.
All experiment registration, private instance generation, query budget enforcement,
and patch evaluation are controlled authoritatively by the isolated evaluator over
a structured stdin/stdout pipe IPC protocol (Prompts 1–5, 8).
"""

from __future__ import annotations

import atexit
import copy
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

import torch
import torch.nn as nn

from weightliftai.weight_programming.defect_families import (
    DefectFamily,
    DiagnosticCondition,
    deserialize_model_state_dict,
)
from weightliftai.weight_programming.evaluator_authority import get_default_ledger_path
from weightliftai.weight_programming.ipc_protocol import (
    DEFAULT_RESOURCE_LIMITS,
    PROTOCOL_VERSION,
    ErrorCategory,
    EvaluatorIPCError,
    verify_signed_evaluation_result,
)
from weightliftai.weight_programming.schemas import PatchBudget, WeightPatch


class EvaluationBudgetExceededError(RuntimeError):
    """Raised when the evaluator rejects a submission due to query budget exhaustion."""


class ProposerAuthorizationError(RuntimeError):
    """Raised when the evaluator rejects an unauthorized, expired, or altered submission."""


class ProposerIsolationViolationError(AttributeError):
    """Raised when an agent attempts to access sealed secrets or out-of-condition data."""


class _IsolatedEvaluatorSubprocessHandle:
    """Manages a single persistent child OS process (`isolated_evaluator.py --ipc-stream`) over stdin/stdout pipes."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[bytes] | None = None

    def _ensure_started(self) -> subprocess.Popen[bytes]:
        if self.proc is not None and self.proc.poll() is None:
            return self.proc
        cmd = [
            sys.executable,
            "-m",
            "weightliftai.weight_programming.isolated_evaluator",
            "--ipc-stream",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return self.proc

    def request(self, raw_request_line: bytes, timeout_seconds: float) -> bytes:
        with self.lock:
            for _attempt in range(2):
                proc = self._ensure_started()
                assert proc.stdin is not None and proc.stdout is not None
                try:
                    proc.stdin.write(raw_request_line)
                    proc.stdin.flush()
                    result_container: list[bytes] = []
                    exc_container: list[BaseException] = []

                    def _read_line() -> None:
                        try:
                            assert proc.stdout is not None
                            result_container.append(proc.stdout.readline())
                        except BaseException as e:
                            exc_container.append(e)

                    reader_thread = threading.Thread(target=_read_line, daemon=True)
                    reader_thread.start()
                    reader_thread.join(timeout=timeout_seconds)
                    if reader_thread.is_alive():
                        self.close()
                        raise EvaluatorIPCError(
                            ErrorCategory.RESOURCE_LIMIT_ERROR,
                            "IPC_SUBPROCESS_TIMEOUT",
                            f"Evaluator subprocess timed out after {timeout_seconds}s.",
                        )
                    if exc_container:
                        self.close()
                        continue
                    out = result_container[0] if result_container else b""
                    if not out:
                        self.close()
                        continue
                    return out
                except EvaluatorIPCError:
                    raise
                except Exception:
                    self.close()
                    continue
            raise EvaluatorIPCError(
                ErrorCategory.INTERNAL_EVALUATOR_ERROR,
                "EMPTY_EVALUATOR_RESPONSE",
                "Evaluator subprocess returned an empty response.",
            )

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=2.0)
            except Exception:
                pass
            self.proc = None


_SINGLETON_WORKER = _IsolatedEvaluatorSubprocessHandle()
atexit.register(_SINGLETON_WORKER.close)


def send_evaluator_ipc_request(
    action: str,
    payload: dict[str, Any],
    ledger_path: Path | str | None = None,
    timeout_seconds: float = 30.0,
    protocol_version: str = PROTOCOL_VERSION,
) -> dict[str, Any]:
    """Send a structured JSON request to the isolated evaluator subprocess over stdin/stdout pipes."""
    resolved_ledger = str((Path(ledger_path) if ledger_path is not None else get_default_ledger_path()).resolve())
    envelope = {
        "protocol_version": protocol_version,
        "action": action,
        "ledger_path": resolved_ledger,
        "payload": payload,
    }
    raw_request_line = (json.dumps(envelope, separators=(",", ":")) + "\n").encode("utf-8")

    stdout_bytes = _SINGLETON_WORKER.request(raw_request_line, timeout_seconds=timeout_seconds).strip()

    if len(stdout_bytes) > DEFAULT_RESOURCE_LIMITS.max_response_bytes:
        raise EvaluatorIPCError(
            ErrorCategory.RESOURCE_LIMIT_ERROR,
            "IPC_RESPONSE_TOO_LARGE",
            "Evaluator response exceeded maximum permitted byte size.",
        )

    try:
        resp = json.loads(stdout_bytes.decode("utf-8"))
    except Exception as exc:
        raise EvaluatorIPCError(
            ErrorCategory.PROTOCOL_ERROR,
            "INVALID_EVALUATOR_JSON",
            "Evaluator returned malformed JSON.",
        ) from exc

    if not resp.get("success", False):
        cat_str = resp.get("error_category", ErrorCategory.INTERNAL_EVALUATOR_ERROR.value)
        code_str = resp.get("error_code", "UNKNOWN_ERROR")
        msg_str = resp.get("message", "Evaluator request failed.")
        try:
            cat = ErrorCategory(cat_str)
        except ValueError:
            cat = ErrorCategory.INTERNAL_EVALUATOR_ERROR
        raise EvaluatorIPCError(cat, code_str, msg_str)

    return resp["payload"]


class BlindProposerEnvironment:
    """Narrow proposer environment backed by evaluator-authoritative registration and IPC."""

    def __init__(
        self,
        seed: int = 42,
        max_queries: int = 1,
        defect_family: str = DefectFamily.FC2_ROW0_ZERO_MASK.value,
        diagnostic_condition: str = DiagnosticCondition.FULL_DIAGNOSTIC_TOOLKIT.value,
        ttl_seconds: float = 3600.0,
        ledger_path: Path | str | None = None,
        _existing_registration: dict[str, Any] | None = None,
    ):
        self.public_seed_hint = seed
        self._untrusted_caller_max_queries = max_queries
        self._queries_used = 0
        self._ledger_path = Path(ledger_path) if ledger_path is not None else None

        if _existing_registration is not None:
            reg = copy.deepcopy(_existing_registration)
        else:
            reg = send_evaluator_ipc_request(
                action="register_experiment",
                payload={
                    "defect_family": defect_family,
                    "diagnostic_condition": diagnostic_condition,
                    "ttl_seconds": ttl_seconds,
                },
                ledger_path=self._ledger_path,
            )

        self._registration_snapshot = copy.deepcopy(reg)
        self.experiment_id: str = reg["experiment_id"]
        self.benchmark_instance_id: str = reg["benchmark_instance_id"]
        self.defect_family: str = reg["defect_family"]
        self.diagnostic_condition: str = reg["diagnostic_condition"]
        self.allowed_submissions: int = int(reg["allowed_submissions"])
        self.attempt_policy: str = reg["attempt_policy"]
        self.expires_at: float = float(reg["expires_at"])
        self.config_sha256: str = reg["config_sha256"]
        self.evaluator_public_key: str = reg["evaluator_public_key"]

        bundle = reg["proposer_bundle"]
        self.corrupted_fingerprint: str = bundle["base_fingerprint"]
        self._parameters_metadata: dict[str, Any] = bundle["parameters"]
        self._diagnostic_payload: dict[str, Any] = bundle["diagnostic_payload"]

        b_cfg = bundle["budget"]
        self.budget: PatchBudget = PatchBudget(
            max_changed_tensors=int(b_cfg["max_changed_tensors"]),
            max_l2_norm=float(b_cfg["max_l2_norm"]),
            max_relative_norm=float(b_cfg["max_relative_norm"]),
            allowed_tensor_names=list(b_cfg["allowed_tensor_names"]),
        )

        if "corrupted_state_dict" in self._diagnostic_payload:
            self._corrupted_model: nn.Module | None = deserialize_model_state_dict(
                self._diagnostic_payload["corrupted_state_dict"]
            )
        else:
            self._corrupted_model = None

    @classmethod
    def from_existing_registration(
        cls,
        env: "BlindProposerEnvironment",
    ) -> "BlindProposerEnvironment":
        """Construct a second environment handle attached to the same registered experiment_id."""
        return cls(
            ledger_path=env._ledger_path,
            _existing_registration=env._registration_snapshot,
        )

    def __getattr__(self, name: str) -> Any:
        if name == "corrupted_model":
            raise ProposerIsolationViolationError(
                f"Access denied: Model weights are not permitted under diagnostic_condition='{self.diagnostic_condition}'."
            )
        forbidden_tokens = ("_sealed", "clean", "private_seed", "defect_mask", "oracle")
        if any(tok in name for tok in forbidden_tokens) or name == "evaluator":
            raise ProposerIsolationViolationError(
                f"Access denied: Proposer environment strictly isolates secrets. Attribute '{name}' does not exist."
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def corrupted_model(self) -> nn.Module:
        if self._corrupted_model is None:
            raise ProposerIsolationViolationError(
                f"Access denied: Model weights are not permitted under diagnostic_condition='{self.diagnostic_condition}'."
            )
        return self._corrupted_model

    def get_corrupted_model(self) -> nn.Module:
        """Return a disposable clone of the corrupted model (if permitted by diagnostic_condition)."""
        if self._corrupted_model is None:
            raise ProposerIsolationViolationError(
                f"Access denied: Model weights are not permitted under diagnostic_condition='{self.diagnostic_condition}'."
            )
        return copy.deepcopy(self._corrupted_model)

    def get_architecture_metadata(self) -> dict[str, Any]:
        """Return parameter names, shapes, and layer information."""
        return {
            "experiment_id": self.experiment_id,
            "benchmark_instance_id": self.benchmark_instance_id,
            "defect_family": self.defect_family,
            "diagnostic_condition": self.diagnostic_condition,
            "architecture": "ToyMLP(4 -> 8 -> 2)",
            "base_fingerprint": self.corrupted_fingerprint,
            "parameters": copy.deepcopy(self._parameters_metadata),
            "budget": {
                "max_changed_tensors": self.budget.max_changed_tensors,
                "max_l2_norm": self.budget.max_l2_norm,
                "max_relative_norm": self.budget.max_relative_norm,
                "allowed_tensor_names": list(self.budget.allowed_tensor_names or []),
            },
        }

    def get_authorized_diagnostic_package(self) -> dict[str, Any]:
        """Return the exact diagnostic payload authorized for this experiment's diagnostic_condition."""
        pkg = copy.deepcopy(self._diagnostic_payload)
        pkg.pop("corrupted_state_dict", None)
        return pkg

    def get_diagnostic_failure_report(self) -> dict[str, Any]:
        """Return empirical error on the diagnostic task (if behavioral failures are permitted)."""
        if "behavioral_failures" not in self._diagnostic_payload:
            raise ProposerIsolationViolationError(
                f"Access denied: Behavioral failure examples are not permitted under diagnostic_condition='{self.diagnostic_condition}'."
            )
        bf = self._diagnostic_payload["behavioral_failures"]
        return {
            "experiment_id": self.experiment_id,
            "diagnostic_condition": self.diagnostic_condition,
            "target_task": self._diagnostic_payload["target_task"],
            "control_task": self._diagnostic_payload["control_task"],
            "observed_diagnostic_mse": float(bf["observed_diagnostic_mse"]),
            "diagnostic_sample_count": int(bf["diagnostic_sample_count"]),
            "inputs": torch.tensor(bf["inputs"], dtype=torch.float32),
            "expected_outputs": torch.tensor(bf["expected_outputs"], dtype=torch.float32),
            "corrupted_predictions": torch.tensor(bf["corrupted_predictions"], dtype=torch.float32),
        }

    def submit_candidate_patch(
        self,
        patch: WeightPatch,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Submit candidate patch to the isolated evaluator over pipe IPC."""
        if patch.base_model_fingerprint != self.corrupted_fingerprint:
            raise ValueError(
                f"Candidate patch fingerprint mismatch: patch declared '{patch.base_model_fingerprint}' "
                f"but active baseline has '{self.corrupted_fingerprint}'."
            )

        patch_dict = {
            "patch_id": patch.patch_id,
            "base_model_id": patch.base_model_id,
            "base_model_revision": patch.base_model_revision,
            "base_model_fingerprint": patch.base_model_fingerprint,
            "deltas": {k: v.tolist() for k, v in patch.deltas.items()},
            "declared_shapes": {k: list(v.shape) for k, v in patch.deltas.items()},
            "declared_dtypes": {k: str(v.dtype) for k, v in patch.deltas.items()},
            "hypothesis": patch.hypothesis,
            "proposed_by": patch.proposed_by,
            "rationale": getattr(patch, "rationale", ""),
        }

        request_payload = {
            "experiment_id": self.experiment_id,
            "config_sha256": self.config_sha256,
            "candidate_patch_sha256": patch.sha256(),
            "patch": patch_dict,
        }

        try:
            result = send_evaluator_ipc_request(
                action="submit_candidate",
                payload=request_payload,
                ledger_path=self._ledger_path,
                timeout_seconds=timeout_seconds,
            )
        except EvaluatorIPCError as ipc_err:
            if ipc_err.code == "QUERY_BUDGET_EXCEEDED":
                raise EvaluationBudgetExceededError(
                    f"Evaluation query budget exceeded for experiment '{self.experiment_id}': {ipc_err.public_message}"
                ) from ipc_err
            if ipc_err.category == ErrorCategory.AUTH_ERROR:
                raise ProposerAuthorizationError(
                    f"[{ipc_err.category.value}:{ipc_err.code}] {ipc_err.public_message}"
                ) from ipc_err
            raise RuntimeError(
                f"[{ipc_err.category.value}:{ipc_err.code}] {ipc_err.public_message}"
            ) from ipc_err

        if not verify_signed_evaluation_result(result, public_key_hex=self.evaluator_public_key):
            raise RuntimeError("Evaluator returned a result with an invalid or missing cryptographic signature.")

        self._queries_used += 1
        return result
