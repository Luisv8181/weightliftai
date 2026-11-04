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

    atlas_parser = subparsers.add_parser(
        "atlas", help="Extract representation layer atlas and compute cross-variant alignment."
    )
    atlas_parser.add_argument("model_id", help="Hugging Face model ID or path.")
    atlas_parser.add_argument("--dataset", help="Path to matched dataset CSV/JSON.")
    atlas_parser.add_argument("--medical-pilot", action="store_true", help="Use built-in 50-concept medical pilot dataset.")
    atlas_parser.add_argument("--split", choices=["all", "train", "val", "test"], default="test", help="Dataset split to evaluate (default: test).")
    atlas_parser.add_argument("--layers", default="samples", help="Comma-separated layer indices, 'all', or 'samples' (5 depths: 0, 25%%, 50%%, 75%%, 100%%).")
    atlas_parser.add_argument("--pooling", choices=["mean", "last", "max"], default="mean", help="Token pooling method (default: mean).")
    atlas_parser.add_argument("--site", choices=["residual", "attention", "mlp"], default="residual", help="Module site to capture (default: residual).")
    atlas_parser.add_argument("--device", default="auto", help="Compute device ('cuda', 'cpu', 'auto').")
    atlas_parser.add_argument("--batch-size", type=int, default=8, help="Batch size for extraction (default: 8).")
    atlas_parser.add_argument("--out-dir", default="atlas_output", help="Directory to save artifacts and HTML report.")
    _add_model_safety_flags(atlas_parser)

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
        elif args.command == "atlas":
            from pathlib import Path
            import torch
            from transformers import AutoTokenizer
            from weightliftai.cartography.layer_atlas import (
                compute_layer_metrics,
                extract_layer_representations,
                find_decoder_layers,
                generate_atlas_html,
            )
            from weightliftai.cartography.medical_pilot import MedicalPilotDataset

            if args.medical_pilot:
                pilot = MedicalPilotDataset()
                df = pilot.to_matched_dataframe(split=None if args.split == "all" else args.split)
                dataset_name = pilot.dataset_ref.dataset_id
            elif args.dataset:
                p = Path(args.dataset)
                if p.suffix.lower() == ".csv":
                    import pandas as pd
                    df = pd.read_csv(p)
                else:
                    import pandas as pd
                    df = pd.read_json(p)
                dataset_name = p.stem
            else:
                parser.error("Either --dataset or --medical-pilot must be specified.")

            if args.device == "auto":
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                dev = torch.device(args.device)

            print(f"Loading tokenizer: {args.model_id}...", file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_id,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print(f"Loading model: {args.model_id} on {dev}...", file=sys.stderr)
            dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
            if "gemma-4" in args.model_id.lower():
                from transformers import Gemma4ForConditionalGeneration
                model = Gemma4ForConditionalGeneration.from_pretrained(
                    args.model_id,
                    torch_dtype=dtype,
                    device_map="auto" if dev.type == "cuda" else None,
                    local_files_only=args.local_files_only,
                    trust_remote_code=args.trust_remote_code,
                )
            else:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    torch_dtype=dtype,
                    device_map="auto" if dev.type == "cuda" else None,
                    local_files_only=args.local_files_only,
                    trust_remote_code=args.trust_remote_code,
                )

            decoder_layers = find_decoder_layers(model)
            total_layers = len(decoder_layers)

            if args.layers == "all":
                layer_indices = list(range(total_layers))
            elif args.layers == "samples":
                q1 = total_layers // 4
                q2 = total_layers // 2
                q3 = (total_layers * 3) // 4
                last = total_layers - 1
                layer_indices = sorted(list({0, q1, q2, q3, last}))
            else:
                layer_indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

            print(f"Extracting representations across {len(layer_indices)} layers for {len(df)} samples...", file=sys.stderr)
            embeddings_by_layer = extract_layer_representations(
                model=model,
                tokenizer=tokenizer,
                texts=df["text"].tolist(),
                layers=layer_indices,
                site=args.site,
                pooling=args.pooling,
                batch_size=args.batch_size,
                device=dev,
            )

            atlas_metrics = {}
            for L in layer_indices:
                m = compute_layer_metrics(embeddings_by_layer[L], df=df)
                m["depth_pct"] = round(L / max(1, total_layers - 1) * 100, 1)
                atlas_metrics[L] = m

            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            metrics_path = out_dir / "layer_metrics.json"
            metrics_path.write_text(json.dumps(atlas_metrics, indent=2), encoding="utf-8")

            html_report = generate_atlas_html(
                model_name=args.model_id,
                dataset_name=f"{dataset_name} ({args.split})",
                layer_metrics=atlas_metrics,
            )
            html_path = out_dir / "layer_atlas.html"
            html_path.write_text(html_report, encoding="utf-8")

            _print({
                "model_id": args.model_id,
                "dataset": dataset_name,
                "split": args.split,
                "sample_count": len(df),
                "layers": layer_indices,
                "metrics_file": str(metrics_path),
                "html_report": str(html_path),
                "trajectory_summary": {
                    L: {
                        "depth_pct": atlas_metrics[L]["depth_pct"],
                        "pairwise_cosines": {
                            p: atlas_metrics[L]["pairwise_cosines"][p]["mean"]
                            for p in atlas_metrics[L]["pairwise_cosines"]
                        },
                    }
                    for L in layer_indices
                },
            })
        elif args.command == "local-dir":
            _print(inspect_local_directory(args.path))
    except Exception as exc:
        parser.exit(2, f"weightliftai: error: {exc}\n")


if __name__ == "__main__":
    main()
