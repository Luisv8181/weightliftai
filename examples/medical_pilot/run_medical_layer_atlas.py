"""Execution pipeline for Medical Terminology Representation Atlas.

Captures multi-layer representations of 50 matched medical concepts across 3 expression variants:
1. canonical_clinical (e.g. "type 2 diabetes mellitus", "cerebrovascular accident")
2. plain_lay (e.g. "adult-onset diabetes", "stroke")
3. cultural_idiom (e.g. "azúcar alta en la sangre", "trombosis / derrame cerebral")

Evaluates whether clinical and cultural expressions converge into an invariant conceptual
manifold at the structural midpoint (50% depth interlingua hub) across:
- google/gemma-3-1b-it (26 layers: 0, 7, 13, 19, 25)
- google/gemma-4-E2B-it (35 layers: 0, 8, 17, 26, 34)

Saves JSON metrics with paired bootstrap 95% CIs and self-contained HTML visual reports.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma4ForConditionalGeneration,
)

from weightliftai.cartography.layer_atlas import (
    compute_layer_metrics,
    extract_layer_representations,
    find_decoder_layers,
    generate_atlas_html,
)
from weightliftai.cartography.medical_pilot import MedicalPilotDataset

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_model_atlas(
    model_id: str,
    layer_indices: list[int],
    df: pd.DataFrame,
    device: str,
    is_gemma4: bool = False,
) -> dict:
    print(f"\n{'='*70}")
    print(f"Beginning Medical Layer Atlas on: {model_id}")
    print(f"Layers to evaluate: {layer_indices}")
    print(f"Dataset items: {len(df)} (50 concepts x 3 variants)")
    print(f"{'='*70}\n")

    t0 = time.time()
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    print(f"Loading tokenizer: {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model: {model_id} on {device} (dtype={dtype})...")
    if is_gemma4:
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()

    decoder_blocks = find_decoder_layers(model)
    total_layers = len(decoder_blocks)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s | Total layers: {total_layers}")

    print("Extracting multi-layer representations in one controlled pass...")
    t_ext = time.time()
    embeddings_by_layer = extract_layer_representations(
        model=model,
        tokenizer=tok,
        texts=df["text"].tolist(),
        layers=layer_indices,
        site="residual",
        token_mask="content",
        pooling="mean",
        batch_size=8,
        device=device,
    )
    ext_time = time.time() - t_ext
    print(f"Extraction completed in {ext_time:.1f}s")

    # Compute paired metrics with bootstrap CIs
    print("Computing pairwise alignment metrics and 2,000-draw bootstrap 95% CIs...")
    layer_metrics = {}
    for L in layer_indices:
        d_pct = round(L / max(1, total_layers - 1) * 100, 1)
        m = compute_layer_metrics(embeddings_by_layer[L], df=df, n_bootstrap=2000, seed=42)
        m["depth_pct"] = d_pct
        layer_metrics[L] = m

        print(f"  Layer {L:02d} ({d_pct:5.1f}% depth):")
        for pair_key, pdata in m["pairwise_cosines"].items():
            mean = pdata["mean"]
            ci = pdata["ci_95"]
            print(f"    - {pair_key:40s}: cos = {mean:.3f} (95% CI [{ci[0]:.3f}, {ci[1]:.3f}])")

    # Generate HTML report
    safe_name = model_id.replace("/", "_").replace("-", "_")
    html_content = generate_atlas_html(
        model_name=model_id,
        dataset_name="NLM MeSH 50-Concept Medical Pilot (Balanced Triad)",
        layer_metrics=layer_metrics,
        title=f"Medical Terminology Layer Atlas — {model_id}",
    )
    html_path = RESULTS_DIR / f"{safe_name}_medical_atlas.html"
    html_path.write_text(html_content, encoding="utf-8")

    # Summary record
    summary = {
        "model_id": model_id,
        "total_layers": total_layers,
        "evaluated_layers": layer_indices,
        "sample_count": len(df),
        "concepts_count": df["item_id"].nunique(),
        "variants": sorted(df["variant"].unique().tolist()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "html_report": str(html_path),
        "layer_metrics": {str(k): v for k, v in layer_metrics.items()},
    }
    json_path = RESULTS_DIR / f"{safe_name}_medical_atlas.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nArtifacts saved:")
    print(f"  - Metrics: {json_path}")
    print(f"  - HTML:    {html_path}")

    # Free memory
    del model
    del tok
    del embeddings_by_layer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main():
    print("Initializing 50-concept Medical Pilot Dataset...")
    ds = MedicalPilotDataset()
    df = ds.to_dataframe()
    print(f"Dataset verified: {len(df)} rows across {df['item_id'].nunique()} MeSH concepts.")
    print("Zero PHI verified. Licensing: MeSH 2026 Public Domain.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute environment: {device} (VRAM: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'})\n")

    # Model 1: Gemma 3 1B IT (26 layers)
    # Midpoint is L13 (52% depth)
    gemma3_layers = [0, 7, 13, 19, 25]
    res_gemma3 = run_model_atlas(
        model_id="google/gemma-3-1b-it",
        layer_indices=gemma3_layers,
        df=df,
        device=device,
        is_gemma4=False,
    )

    # Model 2: Gemma 4 E2B IT (35 layers)
    # Midpoint is L17 (50% depth)
    gemma4_layers = [0, 8, 17, 26, 34]
    res_gemma4 = run_model_atlas(
        model_id="google/gemma-4-E2B-it",
        layer_indices=gemma4_layers,
        df=df,
        device=device,
        is_gemma4=True,
    )

    print("\n" + "="*70)
    print("ALL MEDICAL LAYER ATLAS RUNS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
