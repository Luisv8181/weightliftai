"""Command-line interface for WeightLiftAI."""

from __future__ import annotations

import argparse
import json
import sys

from weightliftai.hardware import get_hardware_report
from weightliftai.inspect_model import inspect_local_directory, inspect_model
from weightliftai.manifest import build_manifest, save_manifest


def _print(data: dict) -> None:
    print(json.dumps(data, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="weightliftai",
        description="Inspect and experiment with open-weight neural networks locally.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("hardware", help="Report CPU, PyTorch, and CUDA information.")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a Hugging Face model.")
    inspect_parser.add_argument("model_id", help="Hugging Face model ID or local model directory.")
    inspect_parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Load model parameters to inspect tensors. May require substantial RAM/VRAM.",
    )
    inspect_parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not access the network; use only cached/local model files.",
    )
    inspect_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code from the model repository. Only use trusted repositories.",
    )
    inspect_parser.add_argument(
        "--manifest",
        help="Optional path to save a JSON provenance manifest.",
    )

    local_parser = subparsers.add_parser("local-dir", help="Summarize files in a local model directory.")
    local_parser.add_argument("path")

    args = parser.parse_args()
    try:
        if args.command == "hardware":
            _print(get_hardware_report())
        elif args.command == "inspect":
            details = inspect_model(
                args.model_id,
                load_weights=args.load_weights,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
            )
            _print(details)
            if args.manifest:
                path = save_manifest(build_manifest(args.model_id, details), args.manifest)
                print(f"\nManifest saved: {path}", file=sys.stderr)
        elif args.command == "local-dir":
            _print(inspect_local_directory(args.path))
    except Exception as exc:
        parser.exit(2, f"weightliftai: error: {exc}\n")


if __name__ == "__main__":
    main()
