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


def _add_model_safety_flags(command: argparse.ArgumentParser) -> None:
    command.add_argument("--local-files-only", action="store_true",
                         help="Use only cached/local model files; do not access the network.")
    command.add_argument("--trust-remote-code", action="store_true",
                         help="Allow custom code from the model repository. Only use trusted repositories.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="weightliftai",
        description="Inspect and experiment with open-weight neural networks locally.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("hardware", help="Report CPU, PyTorch, and CUDA information.")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a Hugging Face model.")
    inspect_parser.add_argument("model_id", help="Hugging Face model ID or local model directory.")
    inspect_parser.add_argument("--load-weights", action="store_true",
                                help="Load model parameters to inspect tensors. May require substantial RAM/VRAM.")
    _add_model_safety_flags(inspect_parser)
    inspect_parser.add_argument("--manifest", help="Optional path to save a JSON provenance manifest.")

    modules_parser = subparsers.add_parser("modules", help="List model leaf modules and exact names.")
    modules_parser.add_argument("model_id")
    modules_parser.add_argument("--contains", help="Only show names containing this substring.")
    _add_model_safety_flags(modules_parser)

    activation_parser = subparsers.add_parser(
        "activations", help="Capture descriptive output statistics for selected modules."
    )
    activation_parser.add_argument("model_id")
    activation_parser.add_argument("--text", required=True, help="Non-sensitive text prompt for one forward pass.")
    activation_parser.add_argument("--module", action="append", required=True,
                                   help="Exact module name; repeat this flag to capture multiple modules.")
    activation_parser.add_argument("--top-k", type=int, default=5,
                                   help="Number of largest absolute activation values to report (default: 5).")
    _add_model_safety_flags(activation_parser)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare module activations side-by-side across two prompts."
    )
    compare_parser.add_argument("model_id")
    compare_parser.add_argument("--prompt-a", required=True, help="First prompt text.")
    compare_parser.add_argument("--prompt-b", required=True, help="Second prompt text.")
    compare_parser.add_argument("--module", action="append", required=True,
                                help="Exact module name; repeat this flag to compare multiple modules.")
    compare_parser.add_argument("--pooling", choices=["mean", "last"], default="mean",
                                help="Token pooling method ('mean' or 'last', default: 'mean').")
    compare_parser.add_argument("--top-k", type=int, default=5,
                                help="Number of largest divergent dimensions to report (default: 5).")
    _add_model_safety_flags(compare_parser)

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
        elif args.command == "modules":
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
                low_cpu_mem_usage=True,
            )
            from weightliftai.activations import list_leaf_modules
            _print({"model_id": args.model_id, "modules": list_leaf_modules(model, contains=args.contains)})
        elif args.command == "activations":
            if args.top_k < 0:
                parser.error("--top-k must be zero or greater.")
            from weightliftai.activations import run_activation_analysis
            _print(run_activation_analysis(
                args.model_id,
                args.text,
                args.module,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
                top_k=args.top_k,
            ))
        elif args.command == "compare":
            if args.top_k < 0:
                parser.error("--top-k must be zero or greater.")
            from weightliftai.activations import run_activation_comparison
            _print(run_activation_comparison(
                args.model_id,
                args.prompt_a,
                args.prompt_b,
                args.module,
                pooling=args.pooling,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
                top_k=args.top_k,
            ))
        elif args.command == "local-dir":
            _print(inspect_local_directory(args.path))
    except Exception as exc:
        parser.exit(2, f"weightliftai: error: {exc}\n")


if __name__ == "__main__":
    main()
