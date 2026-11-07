"""Isolated Evaluator Service for Blind Weight-Repair Benchmark.

Executes in a separated operating system process boundary from the candidate proposer.
Communicates via a structured, versioned stdin/stdout pipe IPC protocol (Prompt 3).
Enforces evaluator-authoritative experiment registration, query budgets, private seed
generation, pre-allocation resource limits, standardized error categories, and Ed25519
cryptographic result signing (Prompts 1, 2, 4, 5).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from weightliftai.weight_programming.evaluator_authority import (
    EvaluatorRegistryAuthority,
    get_default_ledger_path,
)
from weightliftai.weight_programming.ipc_protocol import (
    DEFAULT_RESOURCE_LIMITS,
    ErrorCategory,
    EvaluatorIPCError,
    EvaluatorResourceLimits,
    format_ipc_error,
    format_ipc_success,
    validate_ipc_envelope,
)


def handle_ipc_request_bytes(
    raw_bytes: bytes,
    ledger_path: Path | str | None = None,
    resource_limits: EvaluatorResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> tuple[dict[str, Any], int]:
    """Process a single raw IPC request byte buffer and return (response_dict, exit_code)."""
    try:
        envelope = validate_ipc_envelope(raw_bytes, limits=resource_limits)
        action = envelope["action"]
        payload = envelope["payload"]
        effective_ledger = envelope.get("ledger_path") or ledger_path
        authority = EvaluatorRegistryAuthority(
            ledger_path=effective_ledger,
            resource_limits=resource_limits,
        )

        if action == "register_experiment":
            res_payload = authority.register_experiment(payload)
        elif action == "submit_candidate":
            res_payload = authority.submit_candidate(payload)
        elif action == "verify_receipt":
            res_payload = authority.verify_receipt(payload)
        elif action == "researcher_audit_replay":
            res_payload = authority.researcher_audit_replay(payload)
        else:
            raise EvaluatorIPCError(
                ErrorCategory.PROTOCOL_ERROR,
                "UNKNOWN_IPC_ACTION",
                f"Unsupported IPC action '{action}'.",
            )

        return format_ipc_success(res_payload), 0

    except EvaluatorIPCError as ipc_err:
        return format_ipc_error(ipc_err), 1
    except Exception as exc:
        # Never expose raw stack traces, internal file paths, or secret state to proposer (Prompt 5)
        safe_err = EvaluatorIPCError(
            ErrorCategory.INTERNAL_EVALUATOR_ERROR,
            "UNHANDLED_EVALUATOR_EXCEPTION",
            "Evaluator encountered an internal processing error.",
            internal_detail=f"{type(exc).__name__}: {exc}",
        )
        return format_ipc_error(safe_err), 1


def _run_ipc_stream_loop(default_ledger_path: Path | str | None) -> None:
    """Run a persistent line-delimited JSON IPC service loop in this isolated OS process."""
    for raw_line in sys.stdin.buffer:
        line = raw_line.strip()
        if not line:
            continue
        response, _ = handle_ipc_request_bytes(line, ledger_path=default_ledger_path)
        out_bytes = (json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8")
        sys.stdout.buffer.write(out_bytes)
        sys.stdout.buffer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Evaluator Service (Pipe IPC)")
    parser.add_argument(
        "--ipc-stdin",
        action="store_true",
        help="Read a single structured JSON IPC request from stdin and write response to stdout.",
    )
    parser.add_argument(
        "--ipc-stream",
        action="store_true",
        help="Run a persistent line-delimited JSON IPC service over stdin/stdout in an isolated child process.",
    )
    parser.add_argument(
        "--ledger-path",
        type=str,
        default=None,
        help="Optional evaluator SQLite ledger path override.",
    )
    # Legacy flags retained only to reject unauthorized direct invocation with a structured error
    parser.add_argument("--seed", type=int, default=None, help="Deprecated legacy flag (rejected).")
    parser.add_argument("--patch-json", type=str, default=None, help="Deprecated legacy flag (rejected).")
    args = parser.parse_args()

    ledger_path = args.ledger_path or get_default_ledger_path()

    if (not args.ipc_stdin and not args.ipc_stream) or args.patch_json is not None:
        err = EvaluatorIPCError(
            ErrorCategory.AUTH_ERROR,
            "DIRECT_UNREGISTERED_INVOCATION_FORBIDDEN",
            "Direct evaluator invocation without structured IPC experiment registration is forbidden.",
        )
        sys.stdout.write(json.dumps(format_ipc_error(err)))
        sys.stdout.flush()
        sys.exit(1)

    if args.ipc_stream:
        _run_ipc_stream_loop(ledger_path)
        sys.exit(0)

    max_read = DEFAULT_RESOURCE_LIMITS.max_request_bytes + 1024
    raw_bytes = sys.stdin.buffer.read(max_read)
    response, exit_code = handle_ipc_request_bytes(raw_bytes, ledger_path=ledger_path)
    sys.stdout.write(json.dumps(response))
    sys.stdout.flush()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
