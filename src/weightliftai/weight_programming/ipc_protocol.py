"""Controlled IPC Protocol, Resource Limits, Error Categories, and Ed25519 Receipt Signing.

Implements:
- Prompt 1 (Cryptographically signed aggregate results & independent verification)
- Prompt 3 (Structured pipe-based request/response IPC protocol with versioned schemas)
- Prompt 4 (Pre-allocation input, tensor dimension, NaN/Inf, and resource limits)
- Prompt 5 (Standardized proposer-visible error categories and reduced side channels)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
import numpy as np


PROTOCOL_VERSION = "1.0"
EVALUATOR_VERSION = "2.0.0"


class ErrorCategory(str, Enum):
    """Standardized proposer-visible error categories (Prompt 5)."""

    AUTH_ERROR = "AUTH_ERROR"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    RESOURCE_LIMIT_ERROR = "RESOURCE_LIMIT_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_EVALUATOR_ERROR = "INTERNAL_EVALUATOR_ERROR"


class EvaluatorIPCError(RuntimeError):
    """Structured error carrying a standardized proposer-visible ErrorCategory."""

    def __init__(
        self,
        category: ErrorCategory,
        code: str,
        public_message: str,
        internal_detail: str | None = None,
    ):
        super().__init__(public_message)
        self.category = category
        self.code = code
        self.public_message = public_message
        self.internal_detail = internal_detail or public_message


@dataclass(frozen=True)
class EvaluatorResourceLimits:
    """Hard resource and structural limits enforced before expensive tensor allocation (Prompt 4)."""

    max_request_bytes: int = 2_097_152  # 2 MiB max serialized request
    max_response_bytes: int = 2_097_152  # 2 MiB max serialized response
    max_tensors: int = 16
    max_tensor_ndim: int = 4
    max_tensor_dim_size: int = 4096
    max_total_parameters: int = 262_144
    max_abs_numeric_value: float = 1e6
    max_eval_duration_seconds: float = 15.0
    max_cpu_threads: int = 1
    max_memory_mb: int = 1024
    allowed_dtypes: tuple[str, ...] = ("float32", "float64")


DEFAULT_RESOURCE_LIMITS = EvaluatorResourceLimits()


def generate_ed25519_keypair_hex() -> tuple[str, str]:
    """Generate an Ed25519 (private_key_hex, public_key_hex) keypair."""
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    priv_bytes = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    pub_bytes = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
    return priv_bytes.hex(), pub_bytes.hex()


def canonical_json_bytes(obj: dict[str, Any]) -> bytes:
    """Serialize a dictionary deterministically for cryptographic hashing/signing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_config_sha256(config: dict[str, Any]) -> str:
    """Compute immutable SHA-256 over canonical experiment configuration."""
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def sign_evaluation_receipt(receipt_payload: dict[str, Any], private_key_hex: str) -> dict[str, Any]:
    """Sign an evaluation receipt payload with the evaluator's Ed25519 private key."""
    payload_copy = {k: v for k, v in receipt_payload.items() if k != "signature"}
    priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
    sig = priv.sign(canonical_json_bytes(payload_copy))
    signed = dict(payload_copy)
    signed["signature"] = sig.hex()
    return signed


def verify_signed_evaluation_result(
    evaluation_result: dict[str, Any],
    public_key_hex: str | None = None,
) -> bool:
    """Independently verify a signed evaluation receipt (Prompt 1).

    Returns True iff the receipt is well-formed, matches top-level fields, and its Ed25519
    signature verifies under `public_key_hex` (or the embedded `evaluator_public_key` if not overridden).
    """
    if not isinstance(evaluation_result, dict):
        return False
    receipt = evaluation_result.get("signed_receipt")
    if not isinstance(receipt, dict):
        return False

    sig_hex = receipt.get("signature")
    key_hex = public_key_hex or receipt.get("evaluator_public_key")
    if not isinstance(sig_hex, str) or not isinstance(key_hex, str):
        return False

    required_fields = (
        "experiment_id",
        "candidate_patch_id",
        "candidate_patch_sha256",
        "corrupted_fingerprint",
        "evaluator_version",
        "config_sha256",
        "passed_gates",
        "recommendation",
        "aggregate_scores",
        "issued_at",
        "evaluator_public_key",
    )
    for field in required_fields:
        if field not in receipt:
            return False

    # Cross-check top-level fields against signed receipt so caller cannot mutate outer dict
    if "experiment_id" in evaluation_result and evaluation_result["experiment_id"] != receipt["experiment_id"]:
        return False
    if "patch_id" in evaluation_result and evaluation_result["patch_id"] != receipt["candidate_patch_id"]:
        return False
    if "patch_sha256" in evaluation_result and evaluation_result["patch_sha256"] != receipt["candidate_patch_sha256"]:
        return False
    if "passed_gates" in evaluation_result and evaluation_result["passed_gates"] != receipt["passed_gates"]:
        return False
    if "recommendation" in evaluation_result and evaluation_result["recommendation"] != receipt["recommendation"]:
        return False
    if "aggregate_scores" in evaluation_result and evaluation_result["aggregate_scores"] != receipt["aggregate_scores"]:
        return False

    unsigned = {k: v for k, v in receipt.items() if k != "signature"}
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(key_hex))
        pub.verify(bytes.fromhex(sig_hex), canonical_json_bytes(unsigned))
        return True
    except (ValueError, InvalidSignature):
        return False


def _inspect_nested_list_shape_and_values(
    nested: Any,
    limits: EvaluatorResourceLimits,
    current_depth: int = 0,
) -> tuple[list[int], int]:
    """Recursively inspect nested list dimensions and scalar finiteness BEFORE allocating NumPy arrays."""
    if isinstance(nested, (int, float)) and not isinstance(nested, bool):
        val = float(nested)
        if not math.isfinite(val):
            raise EvaluatorIPCError(
                ErrorCategory.VALIDATION_ERROR,
                "NON_FINITE_SCALAR",
                "Candidate tensor contains non-finite values (NaN or Inf).",
            )
        if abs(val) > limits.max_abs_numeric_value:
            raise EvaluatorIPCError(
                ErrorCategory.RESOURCE_LIMIT_ERROR,
                "NUMERIC_MAGNITUDE_EXCEEDED",
                "Candidate tensor numeric magnitude exceeds allowed bounds.",
            )
        return [], 1

    if not isinstance(nested, list):
        raise EvaluatorIPCError(
            ErrorCategory.VALIDATION_ERROR,
            "INVALID_TENSOR_ELEMENT_TYPE",
            "Candidate tensor elements must be finite numbers in rectangular lists.",
        )

    next_depth = current_depth + 1
    if next_depth > limits.max_tensor_ndim:
        raise EvaluatorIPCError(
            ErrorCategory.RESOURCE_LIMIT_ERROR,
            "MAX_TENSOR_NDIM_EXCEEDED",
            f"Candidate tensor rank exceeds maximum allowed rank ({limits.max_tensor_ndim}).",
        )

    dim_len = len(nested)
    if dim_len == 0 or dim_len > limits.max_tensor_dim_size:
        raise EvaluatorIPCError(
            ErrorCategory.RESOURCE_LIMIT_ERROR,
            "INVALID_TENSOR_DIMENSION_SIZE",
            f"Candidate tensor dimension size must be in [1, {limits.max_tensor_dim_size}].",
        )

    expected_subshape: list[int] | None = None
    total_elems = 0
    for item in nested:
        subshape, subcount = _inspect_nested_list_shape_and_values(item, limits, next_depth)
        if expected_subshape is None:
            expected_subshape = subshape
        elif subshape != expected_subshape:
            raise EvaluatorIPCError(
                ErrorCategory.VALIDATION_ERROR,
                "RAGGED_TENSOR_ARRAY",
                "Candidate tensor payload is ragged (inconsistent sub-array shapes).",
            )
        total_elems += subcount
        if total_elems > limits.max_total_parameters:
            raise EvaluatorIPCError(
                ErrorCategory.RESOURCE_LIMIT_ERROR,
                "MAX_TOTAL_PARAMETERS_EXCEEDED",
                f"Candidate tensor element count exceeds limit ({limits.max_total_parameters}).",
            )

    return [dim_len] + (expected_subshape or []), total_elems


def prevalidate_and_deserialize_deltas(
    raw_deltas: Any,
    declared_shapes: dict[str, list[int]] | None = None,
    declared_dtypes: dict[str, str] | None = None,
    limits: EvaluatorResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> dict[str, np.ndarray]:
    """Validate tensor counts, shapes, dtypes, and scalar values BEFORE allocating NumPy arrays."""
    if not isinstance(raw_deltas, dict) or len(raw_deltas) == 0:
        raise EvaluatorIPCError(
            ErrorCategory.VALIDATION_ERROR,
            "EMPTY_OR_INVALID_DELTAS",
            "Candidate patch must contain a non-empty dictionary of tensor deltas.",
        )

    if len(raw_deltas) > limits.max_tensors:
        raise EvaluatorIPCError(
            ErrorCategory.RESOURCE_LIMIT_ERROR,
            "MAX_TENSORS_EXCEEDED",
            f"Candidate patch modifies {len(raw_deltas)} tensors; limit is {limits.max_tensors}.",
        )

    running_total_params = 0
    validated_arrays: dict[str, np.ndarray] = {}

    for name, raw_values in raw_deltas.items():
        if not isinstance(name, str) or not name or len(name) > 128:
            raise EvaluatorIPCError(
                ErrorCategory.VALIDATION_ERROR,
                "INVALID_TENSOR_NAME",
                "Tensor name must be a non-empty string <= 128 characters.",
            )

        if declared_dtypes and name in declared_dtypes:
            dtype_str = str(declared_dtypes[name])
            if dtype_str not in limits.allowed_dtypes:
                raise EvaluatorIPCError(
                    ErrorCategory.VALIDATION_ERROR,
                    "UNSUPPORTED_TENSOR_DTYPE",
                    f"Unsupported tensor dtype '{dtype_str}'. Allowed: {limits.allowed_dtypes}.",
                )

        if declared_shapes and name in declared_shapes:
            decl_shape = declared_shapes[name]
            if not isinstance(decl_shape, list) or len(decl_shape) == 0 or len(decl_shape) > limits.max_tensor_ndim:
                raise EvaluatorIPCError(
                    ErrorCategory.RESOURCE_LIMIT_ERROR,
                    "INVALID_DECLARED_SHAPE",
                    "Declared tensor shape exceeds rank limits.",
                )
            prod = 1
            for d in decl_shape:
                if not isinstance(d, int) or d <= 0 or d > limits.max_tensor_dim_size:
                    raise EvaluatorIPCError(
                        ErrorCategory.RESOURCE_LIMIT_ERROR,
                        "DECLARED_DIMENSION_LIMIT_EXCEEDED",
                        f"Declared tensor dimension exceeds limit ({limits.max_tensor_dim_size}).",
                    )
                prod *= d
                if prod > limits.max_total_parameters:
                    raise EvaluatorIPCError(
                        ErrorCategory.RESOURCE_LIMIT_ERROR,
                        "DECLARED_PARAMETERS_EXCEEDED",
                        f"Declared tensor size exceeds parameter limit ({limits.max_total_parameters}).",
                    )

        inferred_shape, elem_count = _inspect_nested_list_shape_and_values(raw_values, limits)
        if declared_shapes and name in declared_shapes and inferred_shape != list(declared_shapes[name]):
            raise EvaluatorIPCError(
                ErrorCategory.VALIDATION_ERROR,
                "DECLARED_SHAPE_MISMATCH",
                "Serialized tensor data does not match its declared shape.",
            )

        running_total_params += elem_count
        if running_total_params > limits.max_total_parameters:
            raise EvaluatorIPCError(
                ErrorCategory.RESOURCE_LIMIT_ERROR,
                "MAX_TOTAL_PARAMETERS_EXCEEDED",
                f"Total patch parameters ({running_total_params}) exceed limit ({limits.max_total_parameters}).",
            )

        validated_arrays[name] = np.array(raw_values, dtype=np.float32)

    return validated_arrays


def validate_ipc_envelope(raw_bytes: bytes, limits: EvaluatorResourceLimits = DEFAULT_RESOURCE_LIMITS) -> dict[str, Any]:
    """Validate raw IPC message byte size, JSON structure, and protocol version."""
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise EvaluatorIPCError(
            ErrorCategory.PROTOCOL_ERROR,
            "INVALID_BYTE_STREAM",
            "IPC payload must be a byte stream.",
        )
    if len(raw_bytes) == 0:
        raise EvaluatorIPCError(
            ErrorCategory.PROTOCOL_ERROR,
            "EMPTY_IPC_REQUEST",
            "Empty IPC request received.",
        )
    if len(raw_bytes) > limits.max_request_bytes:
        raise EvaluatorIPCError(
            ErrorCategory.RESOURCE_LIMIT_ERROR,
            "IPC_PAYLOAD_TOO_LARGE",
            f"IPC request size ({len(raw_bytes)} bytes) exceeds limit ({limits.max_request_bytes} bytes).",
        )

    try:
        envelope = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise EvaluatorIPCError(
            ErrorCategory.PROTOCOL_ERROR,
            "MALFORMED_JSON",
            "IPC request is not valid UTF-8 JSON.",
            internal_detail=str(exc),
        ) from exc

    if not isinstance(envelope, dict):
        raise EvaluatorIPCError(
            ErrorCategory.PROTOCOL_ERROR,
            "INVALID_ENVELOPE_TYPE",
            "IPC envelope root must be a JSON object.",
        )

    version = envelope.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise EvaluatorIPCError(
            ErrorCategory.PROTOCOL_ERROR,
            "UNSUPPORTED_PROTOCOL_VERSION",
            f"Unsupported IPC protocol_version '{version}'. Expected '{PROTOCOL_VERSION}'.",
        )

    action = envelope.get("action")
    valid_actions = {"register_experiment", "submit_candidate", "verify_receipt", "researcher_audit_replay"}
    if action not in valid_actions:
        raise EvaluatorIPCError(
            ErrorCategory.PROTOCOL_ERROR,
            "UNKNOWN_IPC_ACTION",
            f"Unsupported IPC action '{action}'.",
        )

    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise EvaluatorIPCError(
            ErrorCategory.PROTOCOL_ERROR,
            "INVALID_PAYLOAD_OBJECT",
            "IPC 'payload' field must be a JSON object.",
        )

    return envelope


def format_ipc_success(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a standardized IPC success envelope."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "status": "ok",
        "success": True,
        "error_category": None,
        "error_code": None,
        "message": "ok",
        "payload": payload,
    }


def format_ipc_error(err: EvaluatorIPCError) -> dict[str, Any]:
    """Build a standardized IPC error envelope without leaking internal stack traces or paths."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "status": "error",
        "success": False,
        "error_category": err.category.value,
        "error_code": err.code,
        "message": err.public_message,
        "payload": {},
    }
